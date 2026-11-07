/* ---------------- compare two reps (two players or two reps) ---------------- */
const cmp = { aId: null, bId: null, A: null, B: null, t: 0, playing: false, speed: 0.5, view: 'overlay', size: 'same', sync: 'phase', paths: true, diffs: true, last: 0, phase: null, vb: null };
const SAMPLE_A = 'model:pro', SAMPLE_B = 'model:faults', CURRENT = 'current';
const isModel = id => typeof id === 'string' && id.startsWith('model:');
const entryCache = new Map();
const mphH = (v, h) => Number.isFinite(v) && v > 0 ? Math.round(v*h*0.0254*2.23694) + ' mph' : '—';

function finishEntry(e) {
  const S = sideMap(e.seq.leadLeft), r = e.res;
  e.span = [r.phases.start, r.phases.end]; e.key = r.phases.key; e.fps = e.seq.fps;
  e.anchor = e.seq.frames[r.phases.start][S.bAn*2];
  e.path = e.seq.frames.map((f, i) => {
    if (e.kind === 'hit') { const b = e.seq.batTips && e.seq.batTips[i]; return b && b.c > 0.2 ? [b.x, b.y] : null; }
    return [f[S.bWr*2], f[S.bWr*2+1]];
  });
  e.hands = e.seq.frames.map(f => e.kind === 'hit' ? [(f[18] + f[20])/2, (f[19] + f[21])/2] : null);
  return e;
}
function buildEntry(id) {
  const kind = st.mode;
  if (id === CURRENT) {
    if (!(st.res && st.res.kind === kind && st.seq)) return null;
    const p = currentPlayer();
    return finishEntry({ id, kind, label: st.name, player: p.name, heightIn: st.viewHeight || st.heightIn, seq: st.seq, res: st.res });
  }
  const ck = kind + '|' + id; if (entryCache.has(ck)) return entryCache.get(ck);
  let e = null;
  if (isModel(id)) {
    const key = id.slice(6), m = MODELS[key]; if (!m || !m.kinds.includes(kind)) return null;
    const seq = buildModel(kind, key);
    e = { id, kind, label: m.label, player: m.label.replace(/ style.*$/, ''), heightIn: 72, seq, res: analyzeMotion(seq, kind), model: m };
  } else {
    const rec = st.records.find(r => r.id === id && r.kind === kind); if (!rec) return null;
    const seq = seqFromRecord(rec);
    e = { id, kind, label: rec.name, player: rec.player || '', heightIn: rec.heightIn || 70, seq, res: analyzeMotion(seq, kind), rec };
  }
  finishEntry(e); entryCache.set(ck, e); return e;
}
function validCmpId(id) {
  if (!id) return false;
  if (isModel(id)) { const m = MODELS[id.slice(6)]; return !!(m && m.kinds.includes(st.mode)); }
  if (id === CURRENT) return !!(st.res && st.res.kind === st.mode && st.source === 'video' && !st.saved);
  return st.records.some(r => r.id === id && r.kind === st.mode);
}
function defaultPair() {
  const recs = recsFor(st.mode);
  if (recs.length >= 2) { const b = recs[0]; const a = recs.find(r => (r.player || '') !== (b.player || '')) || recs[1]; return [a.id, b.id]; }
  if (recs.length === 1 && validCmpId(CURRENT)) return [recs[0].id, CURRENT];
  if (recs.length === 1) return [SAMPLE_A, recs[0].id];
  if (validCmpId(CURRENT)) return [SAMPLE_A, CURRENT];
  return [SAMPLE_A, SAMPLE_B];
}
function cmpNames() {
  const A = cmp.A, B = cmp.B;
  if (A.player && B.player && A.player !== B.player) return [A.player, B.player];
  return ['Rep A', 'Rep B'];
}
const nmOf = e => { const [a, b] = cmpNames(); return e === cmp.A ? a : b; };
const phIdx = (e, name) => (e.res.phases.list.find(p => p[0] === name) || [0, e.key])[1];
const phaseAt = (e, i) => { let nm = e.res.phases.list[0][0]; for (const [n, k] of e.res.phases.list) if (i >= k) nm = n; return nm; };

/* time mapping: master t = seconds relative to A's key event */
const idxA = t => clamp(cmp.A.key + t*cmp.A.fps, cmp.A.span[0], cmp.A.span[1]);
function idxB(t) {
  const A = cmp.A, B = cmp.B;
  const i = cmp.sync === 'phase' ? alignIndex(A.res.phases.list, B.res.phases.list, idxA(t), B.seq.frames.length) : B.key + t*B.fps;
  return clamp(i, B.span[0], B.span[1]);
}
function tRange() {
  const A = cmp.A, B = cmp.B;
  const ra = [(A.span[0] - A.key)/A.fps, (A.span[1] - A.key)/A.fps];
  if (cmp.sync === 'phase') return ra;
  return [Math.min(ra[0], (B.span[0] - B.key)/B.fps), Math.max(ra[1], (B.span[1] - B.key)/B.fps)];
}
const scaleOf = e => cmp.size === 'same' ? 1 : e.heightIn / Math.max(cmp.A.heightIn, cmp.B.heightIn);
const frameOf = (e, i) => lerpFrame(e.seq.frames, clamp(i, e.span[0], e.span[1]));
function placeF(e, f) { const s = scaleOf(e), o = new Float32Array(34); for (let j = 0; j < 17; j++) { o[j*2] = (f[j*2] - e.anchor)*s; o[j*2+1] = f[j*2+1]*s; } return o; }
const placeP = (e, p) => p ? [(p[0] - e.anchor)*scaleOf(e), p[1]*scaleOf(e)] : null;
function computeCmpVB() {
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = 0.02;
  for (const e of [cmp.A, cmp.B]) for (let i = e.span[0]; i <= e.span[1]; i++) {
    const f = placeF(e, e.seq.frames[i]); for (let j = 0; j < 17; j++) { x0 = Math.min(x0, f[j*2]); x1 = Math.max(x1, f[j*2]); y0 = Math.min(y0, f[j*2+1]); y1 = Math.max(y1, f[j*2+1]); }
    const p = placeP(e, e.path[i]); if (p) { x0 = Math.min(x0, p[0]); x1 = Math.max(x1, p[0]); y0 = Math.min(y0, p[1]); y1 = Math.max(y1, p[1]); }
  }
  const pad = 0.1; cmp.vb = { x0: x0 - pad, x1: x1 + pad, y0: y0 - pad, y1: y1 + pad*0.6 };
  cmp.ex = [cmp.A, cmp.B].map(e => { let a = Infinity, b = -Infinity; for (let i = e.span[0]; i <= e.span[1]; i++) { const f = placeF(e, e.seq.frames[i]); for (let j = 0; j < 17; j++) { a = Math.min(a, f[j*2]); b = Math.max(b, f[j*2]); } const p = placeP(e, e.path[i]); if (p) { a = Math.min(a, p[0]); b = Math.max(b, p[0]); } } return [a - pad, b + pad]; });
}
// B's joint that plays the same role (front/back) as A's joint j — a lefty's front knee is the right knee
const roleJ = j => (!cmp.A || !cmp.B || cmp.A.seq.leadLeft === cmp.B.seq.leadLeft || j === 0) ? j : (j % 2 ? j + 1 : j - 1);
function jointDiffs(fa, fb) {
  const ha = [(fa[22] + fa[24])/2, (fa[23] + fa[25])/2], hb = [(fb[22] + fb[24])/2, (fb[23] + fb[25])/2];
  const d = new Float32Array(17); for (let j = 0; j < 17; j++) { const k = roleJ(j); d[j] = Math.hypot((fa[j*2] - ha[0]) - (fb[k*2] - hb[0]), (fa[j*2+1] - ha[1]) - (fb[k*2+1] - hb[1])); }
  return d;
}
const DIFF_J = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], DIFF_T = 0.05;

function drawPath(ctx, e, idx, map, s, col) {
  const a = e.span[0], b = e.span[1], cur = Math.round(idx);
  const line = (pts, from, to, alpha, lw) => { ctx.globalAlpha = alpha; ctx.lineWidth = lw; ctx.beginPath(); let on = false; for (let i = from; i <= to; i++) { const p = placeP(e, pts[i]); if (!p) { on = false; continue; } const q = map(p[0], p[1]); on ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1]); on = true; } ctx.stroke(); };
  ctx.save(); ctx.strokeStyle = col; ctx.lineCap = 'round'; ctx.lineJoin = 'round';
  line(e.path, a, b, 0.22, Math.max(1.5, s*0.006));
  line(e.path, a, cur, 0.85, Math.max(2, s*0.009));
  if (e.kind === 'hit') { ctx.setLineDash([s*0.012, s*0.012]); line(e.hands, a, cur, 0.6, Math.max(1.5, s*0.005)); }
  ctx.restore();
}
function drawCompare() {
  const c = $('cCv'); if (!c || !cmp.A || !cmp.B || !cmp.vb) return;
  const { ctx, w, h, dpr } = canvasCtx(c); ctx.clearRect(0, 0, w, h);
  const A = cmp.A, B = cmp.B, vb = cmp.vb, ia = idxA(cmp.t), ib = idxB(cmp.t);
  const ra = frameOf(A, ia), rb = frameOf(B, ib), fa = placeF(A, ra), fb = placeF(B, rb);
  const diff = jointDiffs(ra, rb);
  const ca = css('--pa'), cb = css('--pb'), cf = css('--fix');
  const panes = cmp.view === 'overlay' ? [{ x: 0, w, ents: [[B, ib, fb, cb], [A, ia, fa, ca]] }] : [{ x: 0, w: w/2, ents: [[A, ia, fa, ca]] }, { x: w/2, w: w/2, ents: [[B, ib, fb, cb]] }];
  const [nA, nB] = cmpNames();
  const wMax = panes.length > 1 ? Math.max(cmp.ex[0][1] - cmp.ex[0][0], cmp.ex[1][1] - cmp.ex[1][0]) : vb.x1 - vb.x0;
  for (const pane of panes) {
    const s = Math.min(pane.w/wMax, h/(vb.y1 - vb.y0));
    const xr = panes.length > 1 ? cmp.ex[pane.ents[0][0] === A ? 0 : 1] : [vb.x0, vb.x1];
    const ox = pane.x + (pane.w - (xr[1] - xr[0])*s)/2 - xr[0]*s, oy = (h - (vb.y1 - vb.y0)*s)/2 - vb.y0*s;
    const map = (x, y) => [ox + x*s, oy + y*s];
    ctx.save(); ctx.beginPath(); ctx.rect(pane.x, 0, pane.w, h); ctx.clip();
    ctx.strokeStyle = 'rgba(233,238,244,.07)'; ctx.lineWidth = dpr; const step = s*0.25;
    for (let x = ((ox % step) + step) % step + pane.x; x < pane.x + pane.w; x += step) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke(); }
    for (let y = oy; y > 0; y -= step) { ctx.beginPath(); ctx.moveTo(pane.x, y); ctx.lineTo(pane.x + pane.w, y); ctx.stroke(); }
    ctx.strokeStyle = 'rgba(233,238,244,.35)'; ctx.lineWidth = 2*dpr; ctx.beginPath(); ctx.moveTo(pane.x, oy); ctx.lineTo(pane.x + pane.w, oy); ctx.stroke();
    if (cmp.paths) for (const [e, idx, , col] of pane.ents) drawPath(ctx, e, idx, map, s, col);
    for (const [e, idx, f, col] of pane.ents) {
      if (e.kind === 'hit') { const p = placeP(e, e.path[Math.round(idx)]), hd = placeP(e, e.hands[Math.round(idx)]); if (p && hd) { const P1 = map(hd[0], hd[1]), Q = map(p[0], p[1]); ctx.save(); ctx.strokeStyle = col; ctx.lineCap = 'round'; ctx.lineWidth = Math.max(3, s*0.02*scaleOf(e)); ctx.beginPath(); ctx.moveTo(P1[0], P1[1]); ctx.lineTo(Q[0], Q[1]); ctx.stroke(); ctx.restore(); } }
      drawSkel(ctx, f, map, { color: col, scale: s*scaleOf(e), leadLeft: e.seq.leadLeft });
    }
    if (cmp.diffs) {
      ctx.save(); ctx.strokeStyle = cf; ctx.lineWidth = Math.max(1.5, s*0.005);
      for (const j of DIFF_J) { if (diff[j] < DIFF_T) continue; const r = Math.max(6*dpr, s*(0.02 + Math.min(0.05, diff[j]*0.3)));
        for (const [e, , f] of pane.ents) { if (pane.ents.length > 1 && e === A) continue; const k = e === B ? roleJ(j) : j; const q = map(f[k*2], f[k*2+1]); ctx.beginPath(); ctx.arc(q[0], q[1], r, 0, Math.PI*2); ctx.stroke(); }
        if (pane.ents.length > 1) { const k = roleJ(j), p = map(fa[j*2], fa[j*2+1]), q = map(fb[k*2], fb[k*2+1]); ctx.setLineDash([3*dpr, 3*dpr]); ctx.beginPath(); ctx.moveTo(p[0], p[1]); ctx.lineTo(q[0], q[1]); ctx.stroke(); ctx.setLineDash([]); }
      }
      ctx.restore();
    }
    if (panes.length > 1) { const e = pane.ents[0][0]; ctx.fillStyle = pane.ents[0][3]; ctx.font = `600 ${12*dpr}px ${css('--f-mono') || 'monospace'}`; ctx.fillText((e === A ? nA : nB).toUpperCase(), pane.x + 10*dpr, h - 12*dpr); }
    ctx.restore();
  }
  if (panes.length > 1) { ctx.strokeStyle = 'rgba(233,238,244,.2)'; ctx.lineWidth = dpr; ctx.beginPath(); ctx.moveTo(w/2, 0); ctx.lineTo(w/2, h); ctx.stroke(); }
  const pa = phaseAt(A, ia), pb = phaseAt(B, ib);
  $('cPhase').textContent = pa === pb ? pa : `${nA}: ${pa} · ${nB}: ${pb}`;
  $('cTime').textContent = (cmp.t >= 0 ? '+' : '') + Math.round(cmp.t*1000) + ' ms';
  const [r0, r1] = tRange(), sc = $('cScrub'), v = Math.round((cmp.t - r0)/(r1 - r0)*1000); if (+sc.value !== v) sc.value = v;
  document.querySelectorAll('#cPhaseChips .chip').forEach(ch => ch.setAttribute('aria-current', ch.dataset.cph === pa ? 'true' : 'false'));
  drawSeqCharts();
}
function compareLoop(ts) {
  if (st.tab !== 'Compare' || !cmp.A) { cmp.last = ts; return; }
  if (cmp.playing) {
    const dt = cmp.last ? Math.min(0.1, (ts - cmp.last)/1000) : 0; cmp.t += dt*cmp.speed;
    const [r0, r1] = tRange(); if (cmp.t > r1 + 0.3) cmp.t = r0;
    drawCompare();
  }
  cmp.last = ts;
}
function cmpPlay(on) { cmp.playing = on; $('cPlay').setAttribute('aria-label', on ? 'Pause' : 'Play'); $('cIco').innerHTML = on ? '<path d="M6 4h4v16H6zM14 4h4v16h-4z"/>' : '<path d="M7 4v16l13-8z"/>'; if (on) { const [r0, r1] = tRange(); if (cmp.t >= r1) cmp.t = r0; } }

/* ---------------- comparison numbers ---------------- */
function anglesAt(e, i) {
  const f = e.seq.frames[clamp(Math.round(i), 0, e.seq.frames.length - 1)], S = sideMap(e.seq.leadLeft);
  const hm = mid(f, 11, 12), sm = mid(f, 5, 6);
  const kin = e.res.kin; const ii = clamp(Math.round(i), 0, kin.pelvisAngle.length - 1);
  return {
    leadKnee: angAt(P(f, S.lHip), P(f, S.lKn), P(f, S.lAn)), backKnee: angAt(P(f, S.bHip), P(f, S.bKn), P(f, S.bAn)),
    leadHip: angAt(P(f, S.lSh), P(f, S.lHip), P(f, S.lKn)), backHip: angAt(P(f, S.bSh), P(f, S.bHip), P(f, S.bKn)),
    leadElbow: angAt(P(f, S.lSh), P(f, S.lEl), P(f, S.lWr)), backElbow: angAt(P(f, S.bSh), P(f, S.bEl), P(f, S.bWr)),
    lean: Math.atan2(sm[0] - hm[0], -(sm[1] - hm[1]))*DEG,
    shTilt: Math.atan2(-(f[S.lSh*2+1] - f[S.bSh*2+1]), Math.abs(f[S.lSh*2] - f[S.bSh*2]) + 1e-6)*DEG,
    sep: kin.pelvisAngle[ii] - kin.torsoAngle[ii],
    headX: (f[0] - hm[0])*100,
    stance: (f[S.lAn*2] - f[S.bAn*2])*100,
  };
}
function angleDefs(kind) {
  const hit = kind === 'hit';
  return [
    { k: 'leadKnee', name: hit ? 'Front knee' : 'Stride knee', unit: '°', tol: 10, hi: 'straighter', lo: 'more bent' },
    { k: 'backKnee', name: hit ? 'Back knee' : 'Drive knee', unit: '°', tol: 12, hi: 'straighter', lo: 'more bent' },
    { k: 'leadHip', name: 'Front hip (trunk to thigh)', unit: '°', tol: 12, hi: 'more open', lo: 'more closed' },
    { k: 'backHip', name: 'Back hip (trunk to thigh)', unit: '°', tol: 12, hi: 'more open', lo: 'more closed' },
    { k: 'leadElbow', name: hit ? 'Front elbow' : 'Glove elbow', unit: '°', tol: 15, hi: 'straighter', lo: 'more bent' },
    { k: 'backElbow', name: hit ? 'Back elbow' : 'Throwing elbow', unit: '°', tol: 15, hi: 'straighter', lo: 'more bent' },
    { k: 'lean', name: hit ? 'Trunk lean (+ toward pitcher)' : 'Trunk lean (+ toward plate)', sl: 'trunk', unit: '°', tol: 7, hi: hit ? 'leaning more toward the pitcher' : 'leaning more toward the plate', lo: hit ? 'leaning more back' : 'more upright' },
    { k: 'shTilt', name: 'Shoulder line (+ front shoulder up)', sl: 'shoulder line', unit: '°', tol: 8, hi: 'with the front shoulder higher', lo: 'with the front shoulder lower' },
    { k: 'sep', name: 'Hip-shoulder separation', unit: '°', tol: 10, hi: 'more', lo: 'less' },
    { k: 'headX', name: 'Head ahead of hips', sl: 'head', unit: '%', tol: 4, hi: 'farther forward over the hips', lo: 'farther back behind the hips' },
    { k: 'stance', name: 'Foot spread', sl: 'foot spread', unit: '%', tol: 6, hi: 'wider', lo: 'narrower' },
  ];
}
const fmtA = (v, u) => !Number.isFinite(v) ? '—' : Math.round(v) + (u === '°' ? '°' : '%');
function biggestDiffs() {
  const A = cmp.A, B = cmp.B, kind = A.kind, [nA, nB] = cmpNames(), items = [];
  const phases = kind === 'hit' ? ['Foot plant', 'Contact'] : ['Foot plant', 'Release'];
  for (const d of angleDefs(kind)) {
    let best = null;
    for (const ph of phases) { const va = anglesAt(A, phIdx(A, ph))[d.k], vb = anglesAt(B, phIdx(B, ph))[d.k], df = vb - va; if (Math.abs(df) > d.tol && (!best || Math.abs(df)/d.tol > best.mag)) best = { mag: Math.abs(df)/d.tol, ph, va, vb, df }; }
    if (!best) continue;
    const amt = Math.round(Math.abs(best.df)) + (d.unit === '°' ? '°' : '% of height');
    const text = d.k === 'sep' ? `${nB} has ${amt} ${best.df > 0 ? 'more' : 'less'} hip-shoulder separation than ${nA}`
      : `${nB}'s ${(d.sl || d.name).toLowerCase()} is ${amt} ${best.df > 0 ? d.hi : d.lo}`;
    items.push({ mag: best.mag, phase: best.ph, title: `${d.sl ? d.sl[0].toUpperCase() + d.sl.slice(1) : d.name} at ${best.ph.toLowerCase()}`, text: `${text} (${fmtA(best.vb, d.unit)} vs ${fmtA(best.va, d.unit)}).` });
  }
  // timing
  const ms = (e, a, b) => (phIdx(e, b) - phIdx(e, a))/e.fps*1000;
  const start = kind === 'hit' ? 'Load' : 'Leg lift', key = A.res.phases.keyName;
  const tA = ms(A, start, key), tB = ms(B, start, key);
  if (Math.abs(tB - tA) > Math.max(60, 0.15*tA)) items.push({ mag: Math.abs(tB - tA)/60, phase: start, title: `Tempo, ${start.toLowerCase()} to ${key.toLowerCase()}`, text: `${nB} takes ${Math.round(Math.abs(tB - tA))} ms ${tB > tA ? 'longer' : 'less'} (${Math.round(tB)} vs ${Math.round(tA)} ms).` });
  const fast = kind === 'hit' ? ['Launch', 'Contact'] : ['Foot plant', 'Release'];
  const fA = ms(A, ...fast), fB = ms(B, ...fast);
  if (Math.abs(fB - fA) > Math.max(25, 0.2*fA)) items.push({ mag: Math.abs(fB - fA)/25, phase: fast[0], title: `${fast[0]} to ${fast[1].toLowerCase()}`, text: `${nB} is ${Math.round(Math.abs(fB - fA))} ms ${fB > fA ? 'slower' : 'quicker'} through the ${kind === 'hit' ? 'swing' : 'arm action'} (${Math.round(fB)} vs ${Math.round(fA)} ms).` });
  // energy order
  const oA = A.res.kin.order.join(' → '), oB = B.res.kin.order.join(' → ');
  if (oA !== oB || A.res.kin.inOrder !== B.res.kin.inOrder) {
    const bad = [A, B].filter(e => !e.res.kin.inOrder).map(nmOf);
    items.push({ mag: 2.6, phase: fast[0], title: 'Energy chain order', text: `${nA}: ${oA}. ${nB}: ${oB}. ${bad.length ? bad.join(' and ') + (bad.length > 1 ? ' are' : ' is') + ' out of order, so speed leaks between segments.' : ''}` });
  }
  // checkpoint metrics
  for (const m of A.res.metrics) {
    const mb = B.res.metrics.find(x => x.key === m.key); if (!mb || !Number.isFinite(m.value) || !Number.isFinite(mb.value) || m.key === 'seq' || m.key === 'ttc') continue;
    const tol = DELTA_TOL[m.key] || m.tol/2, df = mb.value - m.value; if (Math.abs(df) <= 2*tol) continue;
    items.push({ mag: Math.abs(df)/(2*tol), phase: m.phase, title: m.label, text: `${nB}: ${fmtShort(mb.value, m.unit)}, ${nA}: ${fmtShort(m.value, m.unit)}.${mb.status !== 'good' && mb.cue ? ' ' + mb.cue : ''}` });
  }
  return items.sort((a, b) => b.mag - a.mag).slice(0, 6);
}
function segRows() {
  const A = cmp.A, B = cmp.B, names = [];
  for (const s of [...A.res.kin.segs, ...B.res.kin.segs]) if (!names.includes(s.name)) names.push(s.name);
  const peakMs = (e, s) => (s.peak - e.key)/e.fps*1000;
  const spd = (e, s) => /Hips|Torso/.test(s.name) ? Math.round(s.peakVal) + '°/s' : mphH(s.peakVal, e.heightIn);
  return names.map(n => { const sa = A.res.kin.segs.find(s => s.name === n), sb = B.res.kin.segs.find(s => s.name === n); return { n, ta: sa ? peakMs(A, sa) : NaN, tb: sb ? peakMs(B, sb) : NaN, va: sa ? spd(A, sa) : '—', vb: sb ? spd(B, sb) : '—' }; });
}
function drawSeqCharts() {
  const A = cmp.A, B = cmp.B; if (!A) return;
  const lo = Math.max(-800, Math.min((A.span[0] - A.key)/A.fps, (B.span[0] - B.key)/B.fps)*1000), hi = Math.min(450, Math.max((A.span[1] - A.key)/A.fps, (B.span[1] - B.key)/B.fps)*1000);
  [['cSeqA', A, idxA(cmp.t)], ['cSeqB', B, idxB(cmp.t)]].forEach(([id, e, cur]) => {
    const el = $(id); if (!el) return; const { ctx, w, h, dpr } = canvasCtx(el); ctx.clearRect(0, 0, w, h);
    const padL = 8*dpr, padR = 8*dpr, padT = 10*dpr, padB = 20*dpr;
    const X = ms => padL + (ms - lo)/(hi - lo)*(w - padL - padR), Y = v => padT + (1 - v)*(h - padT - padB), msOf = i => (i - e.key)/e.fps*1000;
    ctx.strokeStyle = css('--line'); ctx.lineWidth = dpr; ctx.beginPath(); ctx.moveTo(padL, Y(0)); ctx.lineTo(w - padR, Y(0)); ctx.stroke();
    ctx.fillStyle = css('--ink-2'); ctx.font = `${10*dpr}px ${css('--f-mono') || 'monospace'}`; ctx.textAlign = 'center';
    for (let v = Math.ceil(lo/200)*200; v <= hi; v += 200) { const x = X(v); ctx.fillRect(x - 0.5*dpr, Y(0), dpr, 4*dpr); ctx.fillText(v === 0 ? e.res.phases.keyName : (v > 0 ? '+' : '') + v, x, h - 5*dpr); }
    ctx.setLineDash([4*dpr, 4*dpr]); ctx.beginPath(); ctx.moveTo(X(0), padT - 4*dpr); ctx.lineTo(X(0), Y(0)); ctx.stroke(); ctx.setLineDash([]);
    const cols = segColors(), a = Math.max(e.span[0], Math.round(e.key + lo/1000*e.fps)), b = Math.min(e.span[1], Math.round(e.key + hi/1000*e.fps));
    e.res.kin.segs.forEach((s, si) => {
      const mx = Math.max(1e-6, ...s.series.slice(a, b + 1)); ctx.strokeStyle = cols[si]; ctx.lineWidth = 2*dpr; ctx.beginPath();
      for (let i = a; i <= b; i++) { const x = X(msOf(i)), y = Y(s.series[i]/mx); i === a ? ctx.moveTo(x, y) : ctx.lineTo(x, y); } ctx.stroke();
      if (s.peak >= a && s.peak <= b) { ctx.fillStyle = cols[si]; ctx.beginPath(); ctx.arc(X(msOf(s.peak)), Y(s.series[s.peak]/mx), 4*dpr, 0, Math.PI*2); ctx.fill(); ctx.strokeStyle = css('--surface'); ctx.lineWidth = 2*dpr; ctx.stroke(); }
    });
    ctx.strokeStyle = e === A ? css('--pa') : css('--pb'); ctx.lineWidth = 2*dpr; const x = X(msOf(cur)); ctx.beginPath(); ctx.moveTo(x, padT - 6*dpr); ctx.lineTo(x, Y(0)); ctx.stroke();
  });
}
function timelineHtml() {
  const A = cmp.A, B = cmp.B, ms = (e, i) => (i - e.key)/e.fps*1000;
  let lo = Infinity, hi = -Infinity; for (const e of [A, B]) { const l = e.res.phases.list; lo = Math.min(lo, ms(e, l[0][1])); hi = Math.max(hi, ms(e, l[l.length - 1][1])); }
  const X = v => ((v - lo)/(hi - lo)*100).toFixed(2) + '%';
  const bar = (e, col) => { const l = e.res.phases.list; let segs = ''; for (let k = 0; k < l.length - 1; k++) { const a = ms(e, l[k][1]), b = ms(e, l[k+1][1]); segs += `<span class="tl-seg" style="left:${X(a)};width:calc(${X(b)} - ${X(a)});background:${col};opacity:${k % 2 ? 0.45 : 0.9}" title="${esc(l[k][0])} to ${esc(l[k+1][0])}"></span>`; } return `<div class="tl-row"><span class="tl-name">${esc(nmOf(e))}</span><div class="tl-bar">${segs}<i class="tl-key" style="left:${X(0)}"></i></div></div>`; };
  const stp = hi - lo > 1100 ? 400 : 200; let ticks = ''; for (let v = Math.ceil(lo/stp)*stp; v <= hi; v += stp) { if (v !== 0 && Math.abs(v) < stp) continue; if (v - lo < (hi - lo)*0.06) continue; ticks += `<span style="left:${X(v)}">${v === 0 ? esc(A.res.phases.keyName) : (v > 0 ? '+' : '') + v}</span>`; }
  return `<div class="tl">${bar(A, 'var(--pa)')}${bar(B, 'var(--pb)')}<div class="tl-row"><span class="tl-name"></span><div class="tl-axis">${ticks}</div></div></div>`;
}
function pathRows() {
  const A = cmp.A, B = cmp.B, kind = A.kind;
  const len = (e, pts, a, b) => { let s = 0; for (let i = a + 1; i <= b; i++) if (pts[i] && pts[i-1]) s += Math.hypot(pts[i][0] - pts[i-1][0], pts[i][1] - pts[i-1][1]); return s*100; };
  const rows = [];
  if (kind === 'hit') {
    rows.push(['Hand path, launch to contact', ...[A, B].map(e => Math.round(len(e, e.hands, phIdx(e, 'Launch'), e.key)) + '%')]);
    rows.push(['Barrel path, launch to contact', ...[A, B].map(e => e.seq.batOk ? Math.round(len(e, e.path, phIdx(e, 'Launch'), e.key)) + '% of height' : '—')]);
    rows.push(['Peak hand speed', ...[A, B].map(e => mphH(e.seq.info && e.seq.info.handSpeed, e.heightIn))]);
    rows.push(['Peak barrel speed (estimate)', ...[A, B].map(e => mphH(e.seq.info && e.seq.info.batSpeed, e.heightIn))]);
    rows.push(['Hands height at contact', ...[A, B].map(e => Math.round(-e.hands[e.key][1]*100) + '%')]);
  } else {
    const S = e => sideMap(e.seq.leadLeft);
    rows.push(['Arm path, foot plant to release', ...[A, B].map(e => Math.round(len(e, e.path, phIdx(e, 'Foot plant'), e.key)) + '%')]);
    rows.push(['Release height', ...[A, B].map(e => Math.round(-e.seq.frames[e.key][S(e).bWr*2+1]*100) + '%')]);
    rows.push(['Arm slot at release', ...[A, B].map(e => Number.isFinite(e.seq.info && e.seq.info.armSlot) ? Math.round(e.seq.info.armSlot) + '°' : '—')]);
    rows.push(['Hand speed at release (estimate)', ...[A, B].map(e => mphH(e.seq.info && e.seq.info.handSpeed, e.heightIn))]);
  }
  return rows;
}

/* ---------------- compare view ---------------- */
function cmpOptions(sel) {
  const kind = st.mode, M = MODE[kind], groups = {};
  for (const r of recsFor(kind)) { const g = r.player || 'No player set'; (groups[g] = groups[g] || []).push(r); }
  let html = validCmpId(CURRENT) ? `<option value="${CURRENT}">Open now: ${esc(st.name)} (not saved)</option>` : '';
  for (const [g, list] of Object.entries(groups)) html += `<optgroup label="${esc(g)}">` + list.map(r => `<option value="${esc(r.id)}">${esc(r.name)} · ${r.overall}</option>`).join('') + '</optgroup>';
  html += modelOptions(kind);
  return html;
}
function renderCompare(reset) {
  if (!validCmpId(cmp.aId) || !validCmpId(cmp.bId)) { [cmp.aId, cmp.bId] = defaultPair(); reset = true; }
  if (!validCmpId(cmp.bId) || cmp.bId === cmp.aId) { const [a, b] = defaultPair(); cmp.bId = b === cmp.aId ? a : b; }
  const A = buildEntry(cmp.aId), B = buildEntry(cmp.bId); if (!A || !B) return;
  const changed = !cmp.A || cmp.A !== A || cmp.B !== B; cmp.A = A; cmp.B = B;
  const M = MODE[st.mode];
  $('cmpTitle').textContent = 'Compare two ' + M.nouns;
  for (const [id, v] of [['cmpA', cmp.aId], ['cmpB', cmp.bId]]) { const s = $(id); if (document.activeElement !== s) { s.innerHTML = cmpOptions(); s.value = v; } }
  const samples = isModel(cmp.aId) && isModel(cmp.bId);
  const lr = A.seq.leadLeft !== B.seq.leadLeft;
  const blurbs = [A, B].filter(e => e.model).map(e => `<b>${esc(e.model.label)}:</b> ${esc(e.model.blurb)}`).join('<br>');
  $('cmpNote').innerHTML = (blurbs ? blurbs + (A.model && A.model.lefty || B.model && B.model.lefty ? '<br>Style models are approximations built from well-documented traits of each swing, not motion capture.' : '') + '<br>' : '') + (samples ? `Showing the pro model next to a ${M.noun} with common faults. Save your son's ${M.noun} and yours under each player's name in Analyze, then pick them here, or compare either one to the pro model.`
    : `Both skeletons are scaled to the same height${lr ? ', and the lefty is mirrored so both face the same way' : ''}. ${A.heightIn !== B.heightIn ? `Real size shows ${esc(nmOf(A))} at ${fmtHt(A.heightIn)} and ${esc(nmOf(B))} at ${fmtHt(B.heightIn)}.` : ''}`);
  if (changed || reset) { cmp.t = (phIdx(A, A.res.phases.list[0][0]) - A.key)/A.fps; cmp.phase = A.res.phases.keyName; computeCmpVB(); }
  $('cPhaseChips').innerHTML = `<span class="row-lbl" style="align-self:center;margin-right:4px">Phase</span>` + A.res.phases.list.map(([n]) => `<button class="chip phase" data-cph="${esc(n)}">${esc(n)}</button>`).join('');
  const [nA, nB] = cmpNames();
  $('cLegend').innerHTML = `<span><i style="background:var(--pa)"></i>${esc(nA)}</span><span><i style="background:var(--pb)"></i>${esc(nB)}</span>${cmp.diffs ? '<span><i style="background:var(--fix);height:8px;width:8px;border-radius:50%"></i>Joints over 5% of height apart</span>' : ''}${cmp.paths ? `<span><i style="background:var(--ink-2)"></i>${A.kind === 'hit' ? 'Barrel path, dashes are hand path' : 'Throwing hand path'}</span>` : ''}`;
  renderCompareReport();
  drawCompare();
}
const fmtHt = i => Math.floor(i/12) + '′' + (i % 12) + '″';
function renderCompareReport() {
  const A = cmp.A, B = cmp.B, kind = A.kind, [nA, nB] = cmpNames(), M = MODE[kind];
  const items = biggestDiffs();
  const ph = cmp.phase || A.res.phases.keyName;
  const aA = anglesAt(A, phIdx(A, ph)), aB = anglesAt(B, phIdx(B, ph));
  const segs = segRows(), cols = segColors();
  const order = e => e.res.kin.order.map((n, k) => `${k ? '<span class="muted">→</span>' : ''}<b style="background:${cols[e.res.kin.segs.findIndex(s => s.name === n)]}">${esc(n)}</b>`).join('');
  const gaps = (e) => { const s = e.res.kin.segs; const out = []; for (let k = 0; k < s.length - 1; k++) out.push(Math.round((s[k+1].peak - s[k].peak)/e.fps*1000)); return out; };
  const gA = gaps(A), gB = gaps(B), segNames = A.res.kin.segs.map(s => s.name);
  const phl = A.res.phases.list, ivA = [], ivB = [];
  for (let k = 0; k < phl.length - 1; k++) { const a = phl[k][0], b = phl[k+1][0]; ivA.push((phIdx(A, b) - phIdx(A, a))/A.fps*1000); ivB.push((phIdx(B, b) - phIdx(B, a))/B.fps*1000); }
  const flagT = (x, y) => Math.abs(y - x) > Math.max(30, 0.25*Math.abs(x)) ? (Math.abs(y - x) > Math.max(60, 0.5*Math.abs(x)) ? 'big' : 'off') : '';
  const sgn = v => (Math.round(v) > 0 ? '+' : '') + Math.round(v);
  const head = `<tr><th></th><th style="text-align:right"><span class="who" style="--c:var(--pa)">${esc(nA)}</span></th><th style="text-align:right"><span class="who" style="--c:var(--pb)">${esc(nB)}</span></th><th style="text-align:right">Diff</th></tr>`;
  $('cReport').innerHTML = `
  <div class="card">
    <div class="eyebrow">${esc(nA)} vs ${esc(nB)}</div>
    <h2>What's different</h2>
    ${items.length ? `<ol class="diff-list">${items.map(it => `<li><div><b>${esc(it.title)}</b><p class="small" style="margin-top:2px">${esc(it.text)}</p><button class="linkbtn" data-cph="${esc(it.phase)}">Show at ${esc(it.phase.toLowerCase())}</button></div></li>`).join('')}</ol>` : `<p class="muted small">These two ${M.nouns} are very close. No joint, timing or energy difference stands out.</p>`}
  </div>

  <div class="card">
    <div class="eyebrow">Joint by joint</div>
    <h2>Body position at ${esc(ph.toLowerCase())}</h2>
    <div class="chips" style="margin-bottom:10px">${A.res.phases.list.map(([n]) => `<button class="chip" data-cph="${esc(n)}" aria-pressed="${n === ph}">${esc(n)}</button>`).join('')}</div>
    <div class="tbl"><table class="delta"><thead>${head}</thead><tbody>
    ${angleDefs(kind).map(d => { const va = aA[d.k], vb = aB[d.k], df = vb - va; const cls = Math.abs(df) > 2*d.tol ? 'big' : Math.abs(df) > d.tol ? 'off' : ''; return `<tr class="${cls}"><td>${esc(d.name)}</td><td class="num">${fmtA(va, d.unit)}</td><td class="num">${fmtA(vb, d.unit)}</td><td class="num">${sgn(df)}${d.unit === '°' ? '°' : '%'}</td></tr>`; }).join('')}
    </tbody></table></div>
    <p class="note" style="margin-top:8px">Each player is measured at their own ${esc(ph.toLowerCase())}. Orange and red rows are the biggest gaps. % values are percent of each player's height.</p>
  </div>

  <div class="card">
    <div class="eyebrow">Energy chain</div>
    <h2>How energy moves up the body</h2>
    <p class="muted small">Speed of each segment, lined up on ${esc(A.res.phases.keyName.toLowerCase())}. In a good chain each segment peaks after the one below it, and each one goes faster.</p>
    <div class="seq-pair">
      <div><div class="seq-lbl"><span class="who" style="--c:var(--pa)">${esc(nA)}</span><span class="pill ${A.res.kin.inOrder ? 'good' : 'fix'}">${A.res.kin.inOrder ? 'In order' : 'Out of order'}</span></div><div class="chart-wrap" style="height:130px"><canvas id="cSeqA"></canvas></div><div class="order">${order(A)}</div></div>
      <div><div class="seq-lbl"><span class="who" style="--c:var(--pb)">${esc(nB)}</span><span class="pill ${B.res.kin.inOrder ? 'good' : 'fix'}">${B.res.kin.inOrder ? 'In order' : 'Out of order'}</span></div><div class="chart-wrap" style="height:130px"><canvas id="cSeqB"></canvas></div><div class="order">${order(B)}</div></div>
    </div>
    <div class="tbl" style="margin-top:12px"><table class="delta"><thead><tr><th>Peak</th><th style="text-align:right">${esc(nA)}</th><th style="text-align:right">${esc(nB)}</th><th style="text-align:right">${esc(nA)}</th><th style="text-align:right">${esc(nB)}</th></tr><tr><th></th><th style="text-align:right" colspan="2">ms vs ${esc(A.res.phases.keyName.toLowerCase())}</th><th style="text-align:right" colspan="2">peak speed</th></tr></thead><tbody>
    ${segs.map(r => `<tr class="${Number.isFinite(r.ta) && Number.isFinite(r.tb) ? flagT(r.ta, r.tb) : ''}"><td>${esc(r.n)}</td><td class="num">${Number.isFinite(r.ta) ? sgn(r.ta) : '—'}</td><td class="num">${Number.isFinite(r.tb) ? sgn(r.tb) : '—'}</td><td class="num">${r.va}</td><td class="num">${r.vb}</td></tr>`).join('')}
    </tbody></table></div>
    <h3 style="margin-top:14px">Hand-offs between segments</h3>
    <div class="tbl" style="margin-top:6px"><table class="delta"><thead>${head}</thead><tbody>
    ${gA.map((g, k) => { const gb = gB[k]; const nmx = segNames[k] + ' → ' + segNames[k+1]; const bad = v => v < 0; return `<tr class="${Number.isFinite(gb) && (bad(g) !== bad(gb) || Math.abs(gb - g) > 30) ? 'off' : ''}"><td>${esc(nmx)}</td><td class="num">${g} ms${bad(g) ? ' ⚠' : ''}</td><td class="num">${Number.isFinite(gb) ? gb + ' ms' + (bad(gb) ? ' ⚠' : '') : '—'}</td><td class="num">${Number.isFinite(gb) ? sgn(gb - g) : '—'}</td></tr>`; }).join('')}
    </tbody></table></div>
    <p class="note" style="margin-top:8px">A hand-off is the time from one segment's peak to the next. ⚠ means the upper segment peaked first, so energy leaked. Hip and torso speeds are rotation estimates from the video; mph values use each player's height.</p>
  </div>

  <div class="card">
    <div class="eyebrow">Timing</div>
    <h2>Phase by phase</h2>
    ${timelineHtml()}
    <div class="tbl" style="margin-top:12px"><table class="delta"><thead>${head}</thead><tbody>
    ${ivA.map((v, k) => `<tr class="${flagT(v, ivB[k])}"><td>${esc(phl[k][0])} → ${esc(phl[k+1][0].toLowerCase())}</td><td class="num">${Math.round(v)} ms</td><td class="num">${Math.round(ivB[k])} ms</td><td class="num">${sgn(ivB[k] - v)}</td></tr>`).join('')}
    </tbody></table></div>
    <p class="note" style="margin-top:8px">Bars are lined up on ${esc(A.res.phases.keyName.toLowerCase())}. Switch the player to <b>Real timing</b> to watch the tempo difference instead of matching phases.</p>
  </div>

  <div class="card">
    <div class="eyebrow">${kind === 'hit' ? 'Swing path' : 'Arm path'}</div>
    <h2>${kind === 'hit' ? 'Hands and barrel' : 'Arm action'}</h2>
    <div class="tbl"><table class="delta"><thead><tr><th></th><th style="text-align:right"><span class="who" style="--c:var(--pa)">${esc(nA)}</span></th><th style="text-align:right"><span class="who" style="--c:var(--pb)">${esc(nB)}</span></th></tr></thead><tbody>
    ${pathRows().map(r => `<tr><td>${esc(r[0])}</td><td class="num">${esc(r[1])}</td><td class="num">${esc(r[2])}</td></tr>`).join('')}
    </tbody></table></div>
    <p class="note" style="margin-top:8px">Lengths and heights are percent of each player's height. ${kind === 'hit' ? 'A shorter hand path at the same barrel speed means a quicker, more direct swing.' : 'Release height and arm slot are relative to each player\'s height.'}</p>
  </div>

  <div class="card">
    <div class="eyebrow">Checkpoints</div>
    <h2>Scores side by side</h2>
    <div class="tbl"><table class="delta"><thead>${head}</thead><tbody>
    <tr><td><b>Mechanics score</b></td><td class="num"><b>${A.res.overall}</b></td><td class="num"><b>${B.res.overall}</b></td><td class="num">${sgn(B.res.overall - A.res.overall)}</td></tr>
    ${A.res.metrics.map(m => { const mb = B.res.metrics.find(x => x.key === m.key); if (!mb) return ''; const tol = DELTA_TOL[m.key] || m.tol/2, df = mb.value - m.value; const cls = Math.abs(df) > 2*tol ? 'big' : Math.abs(df) > tol ? 'off' : ''; return `<tr class="${cls}"><td>${esc(m.label)}</td><td class="num"><i class="dot ${m.status}"></i>${fmtShort(m.value, m.unit)}</td><td class="num"><i class="dot ${mb.status}"></i>${fmtShort(mb.value, mb.unit)}</td><td class="num">${fmtNum(df, m.unit) > 0 ? '+' : ''}${fmtShort(df, m.unit)}</td></tr>`; }).join('')}
    </tbody></table></div>
    <p class="note" style="margin-top:8px">Dots show each player's own grade on that check: green on target, amber watch, red fix.</p>
  </div>`;
  requestAnimationFrame(drawSeqCharts);
}
function cmpSeekPhase(name) {
  const A = cmp.A; cmpPlay(false); cmp.t = (phIdx(A, name) - A.key)/A.fps; cmp.phase = name; renderCompareReport(); drawCompare();
}
$('cmpA').onchange = e => { cmp.aId = e.target.value; cmpPlay(false); renderCompare(true); };
$('cmpB').onchange = e => { cmp.bId = e.target.value; cmpPlay(false); renderCompare(true); };
$('cmpSwap').onclick = () => { [cmp.aId, cmp.bId] = [cmp.bId, cmp.aId]; renderCompare(true); };
$('cPlay').onclick = () => cmpPlay(!cmp.playing);
$('cPrev').onclick = () => { cmpPlay(false); cmp.t -= 1/cmp.A.fps; const [r0] = tRange(); cmp.t = Math.max(r0, cmp.t); drawCompare(); };
$('cNext').onclick = () => { cmpPlay(false); cmp.t += 1/cmp.A.fps; const [, r1] = tRange(); cmp.t = Math.min(r1, cmp.t); drawCompare(); };
$('cScrub').oninput = e => { cmpPlay(false); const [r0, r1] = tRange(); cmp.t = r0 + (+e.target.value/1000)*(r1 - r0); drawCompare(); };
document.querySelectorAll('[data-cspeed]').forEach(b => b.onclick = () => { cmp.speed = +b.dataset.cspeed; document.querySelectorAll('[data-cspeed]').forEach(x => x.setAttribute('aria-pressed', x === b)); });
document.querySelectorAll('[data-copt]').forEach(b => b.onclick = () => {
  const [k, v] = b.dataset.copt.split(':');
  if (v) { cmp[k] = v; document.querySelectorAll(`[data-copt^="${k}:"]`).forEach(x => x.setAttribute('aria-pressed', x === b)); }
  else { cmp[k] = !cmp[k]; b.setAttribute('aria-pressed', cmp[k]); }
  if (k === 'size') computeCmpVB();
  if (k === 'sync') { const [r0, r1] = tRange(); cmp.t = clamp(cmp.t, r0, r1); }
  renderCompare();
});
document.addEventListener('click', e => { const p = e.target.closest('[data-cph]'); if (p && st.tab === 'Compare') { cmpSeekPhase(p.dataset.cph); if (p.closest('#cReport')) $('cStage').scrollIntoView({ behavior: 'smooth', block: 'center' }); } });
new ResizeObserver(() => { if (st.tab === 'Compare' && cmp.A) drawCompare(); }).observe($('cStage'));
