/* SwingLab core: pure motion analysis. Works on normalized frames:
   units = body height (H), +x = toward the pitcher / plate (forward), y down, ground at y = 0. */
const NJ = 17;
const JN = ['Nose','L eye','R eye','L ear','R ear','L shoulder','R shoulder','L elbow','R elbow','L wrist','R wrist','L hip','R hip','L knee','R knee','L ankle','R ankle'];
const PAIRS = { head: [[1,2],[3,4]], arms: [[5,6],[7,8],[9,10]], legs: [[11,12],[13,14],[15,16]] };

function sideMap(leadLeft) {
  const L = leadLeft;
  return { lSh: L?5:6, bSh: L?6:5, lEl: L?7:8, bEl: L?8:7, lWr: L?9:10, bWr: L?10:9,
           lHip: L?11:12, bHip: L?12:11, lKn: L?13:14, bKn: L?14:13, lAn: L?15:16, bAn: L?16:15 };
}
const P = (f, j) => [f[j*2], f[j*2+1]];
const mid = (f, a, b) => [(f[a*2]+f[b*2])/2, (f[a*2+1]+f[b*2+1])/2];
const dist = (a, b) => Math.hypot(a[0]-b[0], a[1]-b[1]);
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const DEG = 180/Math.PI;
function angAt(a, b, c) {
  const v1 = [a[0]-b[0], a[1]-b[1]], v2 = [c[0]-b[0], c[1]-b[1]];
  const d = Math.hypot(...v1)*Math.hypot(...v2) || 1e-9;
  return Math.acos(clamp((v1[0]*v2[0]+v1[1]*v2[1])/d, -1, 1))*DEG;
}
function argmax(arr, a = 0, b = arr.length-1) { let bi = a; for (let i = a; i <= b; i++) if (arr[i] > arr[bi]) bi = i; return bi; }
function argmin(arr, a = 0, b = arr.length-1) { let bi = a; for (let i = a; i <= b; i++) if (arr[i] < arr[bi]) bi = i; return bi; }
function percentile(arr, p) { const s = arr.filter(Number.isFinite).slice().sort((a,b)=>a-b); if (!s.length) return NaN; return s[clamp(Math.round((s.length-1)*p), 0, s.length-1)]; }
function gauss1d(arr, sigma) {
  if (sigma <= 0) return arr.slice();
  const r = Math.ceil(sigma*2.5), k = []; let ks = 0;
  for (let i = -r; i <= r; i++) { const w = Math.exp(-i*i/(2*sigma*sigma)); k.push(w); ks += w; }
  return arr.map((_, i) => { let s = 0, ws = 0; for (let j = -r; j <= r; j++) { const ii = i+j; if (ii < 0 || ii >= arr.length || !Number.isFinite(arr[ii])) continue; s += arr[ii]*k[j+r]; ws += k[j+r]; } return ws ? s/ws : arr[i]; });
}
function deriv(arr, fps) { const n = arr.length; return arr.map((_, i) => { const a = arr[Math.max(0,i-1)], b = arr[Math.min(n-1,i+1)]; const dt = (Math.min(n-1,i+1)-Math.max(0,i-1))/fps; return dt ? (b-a)/dt : 0; }); }
function series(frames, fn) { return frames.map(fn); }

/* ---------- raw pixel keypoint cleanup ---------- */
// raw: array of Float32Array(51) [x,y,score]*17 in pixels
function cleanRaw(raw, sigma = 1.0) {
  const n = raw.length; if (!n) return raw;
  const out = raw.map(r => Float32Array.from(r));
  // 1) left/right swap fixes per group, against the previous frame
  for (let i = 1; i < n; i++) {
    const prev = out[i-1], cur = out[i];
    for (const g of Object.values(PAIRS)) {
      let keep = 0, swap = 0;
      for (const [a,b] of g) {
        keep += Math.hypot(cur[a*3]-prev[a*3], cur[a*3+1]-prev[a*3+1]) + Math.hypot(cur[b*3]-prev[b*3], cur[b*3+1]-prev[b*3+1]);
        swap += Math.hypot(cur[b*3]-prev[a*3], cur[b*3+1]-prev[a*3+1]) + Math.hypot(cur[a*3]-prev[b*3], cur[a*3+1]-prev[b*3+1]);
      }
      if (swap < keep*0.6) for (const [a,b] of g) for (let c = 0; c < 3; c++) { const t = cur[a*3+c]; cur[a*3+c] = cur[b*3+c]; cur[b*3+c] = t; }
    }
  }
  // 1b) reject one-frame misdetections: a joint far from the median of its neighbors is dropped
  const Hest = percentile(out.map(f => Math.max(f[15*3+1], f[16*3+1]) - f[1]), 0.8) || 100;
  const bad = out.map(() => new Uint8Array(NJ));
  for (let j = 0; j < NJ; j++) for (let i = 0; i < n; i++) {
    const xs = [], ys = [];
    for (let d = -2; d <= 2; d++) { const ii = i + d; if (!d || ii < 0 || ii >= n || out[ii][j*3+2] < 0.2) continue; xs.push(out[ii][j*3]); ys.push(out[ii][j*3+1]); }
    if (xs.length < 2) continue;
    xs.sort((a, b) => a - b); ys.sort((a, b) => a - b);
    const mx = xs[xs.length >> 1], my = ys[ys.length >> 1];
    if (Math.hypot(out[i][j*3] - mx, out[i][j*3+1] - my) > 0.15*Hest) bad[i][j] = 1;
  }
  out.forEach((f, i) => { for (let j = 0; j < NJ; j++) if (bad[i][j]) f[j*3+2] = 0; });
  // 2) interpolate low-confidence joints, 3) smooth
  for (let j = 0; j < NJ; j++) {
    for (const c of [0,1]) {
      let s = out.map(f => f[j*3+2] >= 0.2 ? f[j*3+c] : NaN);
      const good = s.map((v,i)=>Number.isFinite(v)?i:-1).filter(i=>i>=0);
      if (!good.length) continue;
      s = s.map((v, i) => {
        if (Number.isFinite(v)) return v;
        let a = -1, b = -1; for (const g of good) { if (g < i) a = g; else { b = g; break; } }
        if (a < 0) return s[b]; if (b < 0) return s[a];
        return s[a] + (s[b]-s[a])*(i-a)/(b-a);
      });
      s = gauss1d(s, sigma);
      s.forEach((v, i) => out[i][j*3+c] = v);
    }
  }
  return out;
}

/* ---------- normalize pixel frames into body-height units ---------- */
function normalize(clean, leadLeft) {
  const n = clean.length;
  const lens = clean.map(f => { const ny = f[1], ay = (f[15*3+1]+f[16*3+1])/2; return ay - ny; });
  const H = (percentile(lens, 0.8) || 1)/0.88;
  const groundY = percentile(clean.slice(0, Math.max(3, Math.round(n*0.3))).map(f => Math.max(f[15*3+1], f[16*3+1])), 0.5);
  const originX = (clean[0][11*3]+clean[0][12*3])/2;
  // detect forward direction and lead side from ankle travel
  let S = sideMap(leadLeft);
  const travel = j => { const x0 = clean[0][j*3]; let best = 0; for (const f of clean) { const d = f[j*3]-x0; if (Math.abs(d) > Math.abs(best)) best = d; } return best; };
  let tl = travel(S.lAn), tb = travel(S.bAn), autoSwapped = false;
  if (Math.abs(tl) < 0.4*Math.abs(tb) && Math.abs(tb) > 0.05*H) { leadLeft = !leadLeft; S = sideMap(leadLeft); [tl, tb] = [tb, tl]; autoSwapped = true; }
  let fwd = Math.abs(tl) > 0.03*H ? Math.sign(tl) : Math.sign(clean[0][S.lAn*3] - clean[0][S.bAn*3]) || 1;
  const frames = clean.map(f => { const o = new Float32Array(NJ*2); for (let j = 0; j < NJ; j++) { o[j*2] = (f[j*3]-originX)*fwd/H; o[j*2+1] = (f[j*3+1]-groundY)/H; } return o; });
  const conf = clean.map(f => { const o = new Float32Array(NJ); for (let j = 0; j < NJ; j++) o[j] = f[j*3+2]; return o; });
  return { frames, conf, H, groundY, originX, fwd, leadLeft, autoSwapped };
}
function toPixel(seq, x, y) { return [seq.originX + x*seq.fwd*seq.H, seq.groundY + y*seq.H]; }

/* ---------- bat path from per-frame ray scores (dynamic programming) ---------- */
const NA = 72;
function solveBat(scores, allow) {
  // scores: array of Float32Array(NA) or null; allow: per-frame max turn in 5-degree steps (from hand speed)
  const n = scores.length, maxStep = 18;
  const norm = scores.map(s => { if (!s) return null; const m = percentile(Array.from(s), 0.5), mx = Math.max(...s); const den = (mx-m) || 1; return { v: Array.from(s, x => (x-m)/den), conf: clamp((mx-m)/(m+8), 0, 1) }; });
  const cost = [], back = [];
  for (let i = 0; i < n; i++) {
    cost.push(new Float32Array(NA)); back.push(new Int16Array(NA));
    for (let a = 0; a < NA; a++) {
      const u = norm[i] ? -norm[i].v[a]*norm[i].conf : 0;
      if (i === 0) { cost[i][a] = u; continue; }
      let best = Infinity, bi = 0;
      const al = allow ? Math.max(2, Math.min(maxStep, allow[i])) : 8, lim = Math.min(maxStep, Math.ceil(al*1.6));
      for (let d = -lim; d <= lim; d++) { const p = (a+d+NA)%NA; const c = cost[i-1][p] + 0.5*(d/al)**2; if (c < best) { best = c; bi = p; } }
      cost[i][a] = best + u; back[i][a] = bi;
    }
  }
  const path = new Array(n); path[n-1] = argmin(Array.from(cost[n-1]));
  for (let i = n-1; i > 0; i--) path[i-1] = back[i][path[i]];
  // angle is in pixel space; convert direction into normalized space later by caller
  return path.map((a, i) => ({ angleIdx: a, conf: norm[i] ? norm[i].conf*clamp(norm[i].v[a], 0, 1) : 0 }));
}

/* bat path v2: state = (previous angle, angle), so the solver knows the bat's turning speed and keeps it smooth:
   the bat speeds up and slows down gradually and does not flip back and forth. anchors = frames the user tapped. */
function solveBat2(scores, allow, anchors) {
  const n = scores.length, A = NA, wrap = d => ((d + A/2) % A + A) % A - A/2;
  const unary = scores.map((s, i) => {
    const u = new Float32Array(A);
    if (s) { const arr = Array.from(s), m = percentile(arr, 0.5), mx = Math.max(...arr), den = (mx - m) || 1, conf = clamp((mx - m)/(m + 8), 0, 1); for (let a = 0; a < A; a++) u[a] = -((s[a] - m)/den)*conf; }
    const an = anchors && anchors[i];
    if (an != null) for (let a = 0; a < A; a++) { const d = Math.abs(wrap(a - an)); u[a] += d <= 1 ? -6 : 8 + d*0.2; }
    return u;
  });
  let cost = new Float32Array(A*A).fill(Infinity), limPrev = 0;
  for (let a = 0; a < A; a++) cost[a*A + a] = unary[0][a];
  const backs = [null];
  for (let i = 1; i < n; i++) {
    const al = Math.max(1.5, Math.min(16, allow ? allow[i] : 8)), lim = Math.min(18, Math.ceil(al*1.8)), acc = Math.max(1.2, al*0.6);
    // how fast the bat should be turning right now, from how fast the hands move (direction unknown, speed known)
    const e = Math.max(0, (al - 2)*0.85), sig = 1.5 + 0.35*e;
    const nc = new Float32Array(A*A).fill(Infinity), bk = new Int8Array(A*A);
    for (let p = 0; p < A; p++) {
      for (let dq = -limPrev; dq <= limPrev; dq++) {
        const q = (p - dq + A) % A, c0 = cost[q*A + p]; if (c0 === Infinity) continue;
        for (let d = -lim; d <= lim; d++) {
          const a = (p + d + A) % A, k = p*A + a;
          const v = c0 + 0.5*(d/al)**2 + 0.6*((d - dq)/acc)**2 + 0.45*((Math.abs(d) - e)/sig)**2;
          if (v < nc[k]) { nc[k] = v; bk[k] = q; }
        }
      }
      for (let d = -lim; d <= lim; d++) { const a = (p + d + A) % A, k = p*A + a; if (nc[k] !== Infinity) nc[k] += unary[i][a]; }
    }
    cost = nc; backs.push(bk); limPrev = lim;
  }
  let best = 0; for (let k = 1; k < A*A; k++) if (cost[k] < cost[best]) best = k;
  const path = new Array(n); let p = Math.floor(best/A), a = best % A;
  for (let i = n - 1; i >= 1; i--) { path[i] = a; const q = backs[i][p*A + a]; a = p; p = q; }
  path[0] = a;
  return path.map((ai, i) => {
    const s = scores[i]; let conf = 0;
    if (anchors && anchors[i] != null) conf = 1;
    else if (s) { const arr = Array.from(s), m = percentile(arr, 0.5), mx = Math.max(...arr), den = (mx - m) || 1; conf = clamp((mx - m)/(m + 8), 0, 1)*clamp((s[ai] - m)/den, 0, 1); }
    return { angleIdx: ai, conf: Math.max(conf, 0.25) };
  });
}

/* ---------- phase detection ---------- */
function speedSeries(frames, fps, fn) { const p = frames.map(fn); const vx = deriv(p.map(q=>q[0]), fps), vy = deriv(p.map(q=>q[1]), fps); return gauss1d(vx.map((v,i)=>Math.hypot(v, vy[i])), 0.8); }
function rotationSeries(frames, fps, a, b, refEnd) {
  const w = frames.map(f => Math.abs(f[a*2]-f[b*2]));
  const w0 = Math.max(1e-3, percentile(w.slice(0, Math.max(2, refEnd)), 0.9));
  const mi = argmin(w, Math.max(0, Math.floor(refEnd*0.5)), w.length-1);
  // width alone can't tell a coil (turned away from the pitcher) from opening, so use the order of events:
  // anything before the last moment the segment is square again (widest, just before the big turn) is coil → negative
  const ws = gauss1d(w, 1.5); let pk = mi; while (pk > 0 && ws[pk-1] >= ws[pk] - 1e-6) pk--; // walk back from the big turn to the square moment
  let cmin = Infinity; for (let i = 0; i < pk; i++) cmin = Math.min(cmin, ws[i]);
  if (!(cmin < 0.96*ws[pk])) pk = 0; // no real coil before it
  const th = w.map((v, i) => { const t = Math.acos(clamp(v/w0, 0, 1))*DEG; return i > mi ? 180 - t : i < pk ? -t : t; });
  const sm = gauss1d(th, 1.0);
  return { angle: sm, speed: gauss1d(deriv(sm, fps).map(Math.abs), 0.8) };
}

function detectPhases(seq, kind) {
  if (seq.fixedPhases) return seq.fixedPhases;
  const { frames, fps } = seq; const S = sideMap(seq.leadLeft); const n = frames.length; const sec = s => Math.round(s*fps);
  const ankX = frames.map(f => f[S.lAn*2]);
  if (kind === 'hit') {
    const hands = speedSeries(frames, fps, f => mid(f, 9, 10));
    let C = argmax(hands, Math.min(n-1, sec(0.2)), Math.max(0, n-1-sec(0.05)));
    if (seq.batSpeed) { const bs = seq.batSpeed; const b = argmax(bs, Math.max(0, C-3), Math.min(n-1, C+sec(0.12))); if (bs[b] > 0) C = b; } else C = Math.min(n-1, C+1);
    const x0 = ankX[0]; let D = 0; for (let i = 0; i <= C; i++) D = Math.max(D, ankX[i]-x0);
    let FP = -1, S0 = 0;
    if (D > 0.04) { for (let i = 0; i <= C; i++) if (ankX[i]-x0 >= 0.9*D) { FP = i; break; } for (let i = 0; i <= FP; i++) if (ankX[i]-x0 > 0.12*D) { S0 = i; break; } }
    if (FP < 0 || C - FP < 2) { FP = Math.max(0, C - sec(0.2)); S0 = Math.max(0, FP - sec(0.25)); }
    const hx = frames.map(f => mid(f, 9, 10)[0]);
    const L = argmin(hx, Math.max(0, S0 - sec(0.4)), Math.max(Math.max(0, S0 - sec(0.4)), FP));
    const vx = gauss1d(deriv(hx, fps), 0.8); const pk = Math.max(...vx.slice(L, C+1));
    let Lc = FP; for (let i = L; i <= C; i++) if (vx[i] > 0.25*pk) { Lc = Math.max(i, L); break; }
    if (Lc > C) Lc = C;
    const E = Math.min(n-1, argmax(hx, C, Math.min(n-1, C + sec(0.25))));
    const F = Math.min(n-1, C + sec(0.6));
    const ST = Math.max(0, Math.min(L, S0) - sec(0.3));
    return { list: [ ['Stance', ST], ['Load', L], ['Foot plant', FP], ['Launch', Lc], ['Contact', C], ['Extension', E], ['Finish', F] ], start: ST, end: F, key: C, keyName: 'Contact' };
  } else {
    const wr = speedSeries(frames, fps, f => P(f, S.bWr));
    const R = argmax(wr, Math.min(n-1, sec(0.3)), Math.max(0, n-1-sec(0.05)));
    const knY = frames.map(f => f[S.lKn*2+1]);
    const K = argmin(knY, 0, Math.max(0, R - sec(0.15)));
    const x0 = ankX[K]; let D = 0; for (let i = K; i <= R; i++) D = Math.max(D, ankX[i]-x0);
    let FP = -1; for (let i = K; i <= R; i++) if (ankX[i]-x0 >= 0.92*D) { FP = i; break; }
    if (FP < 0 || FP >= R) FP = Math.max(K, R - sec(0.12));
    const rel = frames.map(f => f[S.bWr*2] - f[S.bSh*2]);
    const MER = argmin(rel, FP, Math.max(FP, R-1));
    const F = Math.min(n-1, R + sec(0.35));
    const ST = Math.max(0, K - sec(0.6));
    return { list: [ ['Set', ST], ['Leg lift', K], ['Foot plant', FP], ['Arm cocked', MER], ['Release', R], ['Follow-through', F] ], start: ST, end: F, key: R, keyName: 'Release' };
  }
}
const ph = (phases, name) => (phases.list.find(p => p[0] === name) || [null, phases.key])[1];

/* ---------- kinematic sequence (energy flow) ---------- */
function kinematicSequence(seq, kind, phases) {
  const { frames, fps } = seq; const S = sideMap(seq.leadLeft);
  const fp = ph(phases, 'Foot plant');
  const pelvis = rotationSeries(frames, fps, 11, 12, Math.max(3, fp));
  const torso = rotationSeries(frames, fps, 5, 6, Math.max(3, fp));
  const segs = kind === 'hit'
    ? [ ['Hips', pelvis.speed], ['Torso', torso.speed], ['Hands', speedSeries(frames, fps, f => mid(f, 9, 10))] ]
    : [ ['Hips', pelvis.speed], ['Torso', torso.speed], ['Elbow', speedSeries(frames, fps, f => P(f, S.bEl))], ['Hand', speedSeries(frames, fps, f => P(f, S.bWr))] ];
  if (kind === 'hit' && seq.batSpeed && seq.batOk) segs.push(['Bat', seq.batSpeed]);
  const a = Math.max(0, fp - Math.round(0.25*fps)), b = Math.min(frames.length-1, phases.key + Math.round(0.08*fps));
  const out = segs.map(([name, s]) => { const pk = argmax(s, a, b); return { name, series: s, peak: pk, peakVal: s[pk] }; });
  let ok = 0, tot = 0; for (let i = 0; i < out.length; i++) for (let j = i+1; j < out.length; j++) { tot++; if (out[i].peak <= out[j].peak) ok++; }
  const order = out.slice().sort((x, y) => x.peak - y.peak || out.indexOf(x) - out.indexOf(y)).map(s => s.name);
  return { segs: out, order, score: tot ? ok/tot : 1, inOrder: ok === tot, pelvisAngle: pelvis.angle, torsoAngle: torso.angle };
}

/* ---------- corrections toolkit ---------- */
const HEAD = [0,1,2,3,4];
function bump(i, a, b, c, d) { if (i <= a || i >= d) return 0; if (i < b) return sm01((i-a)/Math.max(1,b-a)); if (i <= c) return 1; return 1 - sm01((i-c)/Math.max(1,d-c)); }
function sm01(t) { t = clamp(t, 0, 1); return t*t*(3-2*t); }
function shiftJ(f, js, dx, dy, w = 1) { for (const j of js) { f[j*2] += dx*w; f[j*2+1] += dy*w; } }
function rotJ(f, js, px, py, deg) { const r = deg/DEG, c = Math.cos(r), s = Math.sin(r); for (const j of js) { const x = f[j*2]-px, y = f[j*2+1]-py; f[j*2] = px + x*c - y*s; f[j*2+1] = py + x*s + y*c; } }
function signedAng(from, to) { let d = Math.atan2(to[1], to[0]) - Math.atan2(from[1], from[0]); while (d > Math.PI) d -= 2*Math.PI; while (d < -Math.PI) d += 2*Math.PI; return d*DEG; }
// straighten/bend a knee to target angle keeping knee+ankle planted: rotate thigh about knee, carry body with hip
function kneeTo(f, S, side, target, w, carryBody) {
  const hip = side === 'l' ? S.lHip : S.bHip, kn = side === 'l' ? S.lKn : S.bKn, an = side === 'l' ? S.lAn : S.bAn;
  const cur = angAt(P(f, hip), P(f, kn), P(f, an)); const delta = (target - cur)*w; if (Math.abs(delta) < 0.5) return;
  const K = P(f, kn), A = P(f, an), Hh = P(f, hip);
  const sgn = Math.sign(signedAng([A[0]-K[0], A[1]-K[1]], [Hh[0]-K[0], Hh[1]-K[1]])) || 1;
  const before = P(f, hip); rotJ(f, [hip], K[0], K[1], sgn*delta); const after = P(f, hip);
  const dx = after[0]-before[0], dy = after[1]-before[1];
  if (carryBody) { const other = side === 'l' ? S.bHip : S.lHip; shiftJ(f, [...HEAD, 5,6,7,8,9,10, other], dx, dy, 1); shiftJ(f, [side === 'l' ? S.bKn : S.lKn], dx, dy, 0.5); }
}

/* ---------- metric definitions ---------- */
function scoreMetric(m) {
  if (!Number.isFinite(m.value)) return null;
  if (m.value >= m.lo && m.value <= m.hi) return 100;
  const d = m.value < m.lo ? m.lo - m.value : m.value - m.hi;
  return Math.max(0, Math.round(100 - 55*d/m.tol));
}

function analyzeMotion(seq, kind) {
  const S = sideMap(seq.leadLeft); const F = seq.frames; const fps = seq.fps;
  const phases = detectPhases(seq, kind);
  const kin = kinematicSequence(seq, kind, phases);
  const g = name => ph(phases, name);
  const at = i => F[clamp(i, 0, F.length-1)];
  const metrics = [];
  const add = m => { m.score = scoreMetric(m); metrics.push(m); };
  const hipMid = f => mid(f, 11, 12), shMid = f => mid(f, 5, 6);
  const tiltBack = f => { const h = hipMid(f), s = shMid(f); return Math.atan2(-(s[0]-h[0]), -(s[1]-h[1]))*DEG; };
  const n = F.length, sec = s => Math.round(s*fps);

  if (kind === 'hit') {
    const ST = g('Stance'), L = g('Load'), FP = g('Foot plant'), LA = g('Launch'), C = g('Contact'), FI = g('Finish');
    const stride = at(FP)[S.lAn*2] - at(ST)[S.lAn*2];
    add({ key: 'stride', label: 'Stride length (overstriding)', value: stride*100, unit: '% ht', lo: 6, hi: 28, tol: 10, weight: 1, phase: 'Foot plant',
      low: 'Little forward move. Without a stride you lose momentum into the ball.', high: 'Overstriding. A long stride drops your eyes and makes you late.',
      drillLow: 'Step-and-hold drill: stride to a target 4 inches ahead, freeze, then swing.', drillHigh: 'Put a bat on the ground as a stride marker and land before it.',
      fix: (f, i) => { const tgt = clamp(stride*100, 6, 28)/100; const dx = tgt - stride; const w = bump(i, g('Load'), FP, n, n+1); shiftJ(f, [S.lAn], dx, 0, w); shiftJ(f, [S.lKn], dx, 0, w*0.7); shiftJ(f, [S.lHip], dx, 0, w*0.25); } });
    const nL = P(at(L), 0), nC = P(at(C), 0);
    const headMove = dist(nL, nC)*100;
    add({ key: 'head', label: 'Head movement, load to contact', value: headMove, unit: '% ht', lo: 0, hi: 6, tol: 5, weight: 1.4, phase: 'Contact',
      high: `Your head travels ${headMove.toFixed(0)}% of your height ${nC[1]-nL[1] > Math.abs(nC[0]-nL[0]) ? 'down' : (nC[0] > nL[0] ? 'toward the pitcher' : 'backward')}. A moving head blurs the ball.`,
      drillHigh: 'Tee work with a partner holding a hand above your cap: keep your head under it through contact.',
      fix: (f, i) => { const w = bump(i, L, Math.round((L+C)/2), FI, n+1); const nose = P(f, 0); const dx = (nL[0]-nose[0]), dy = (nL[1]-nose[1]); const lim = 0.04; const mag = Math.hypot(dx, dy); if (mag < lim) return; const k = (mag - lim)/mag; shiftJ(f, HEAD, dx*k, dy*k, w); shiftJ(f, [5,6], dx*k, dy*k, w*0.75); shiftJ(f, [7,8,9,10], dx*k, dy*k, w*0.6); shiftJ(f, [11,12], dx*k, dy*k, w*0.3); } });
    const loadD = (mid(at(ST), 9, 10)[0] - mid(at(L), 9, 10)[0])*100;
    add({ key: 'load', label: 'Hand load (hands back)', value: loadD, unit: '% ht', lo: 2, hi: 14, tol: 4, weight: 0.8, phase: 'Load',
      low: 'Hands barely move back. Without a load there is no stretch to swing from.', high: 'Hands drift far back, which lengthens the swing.',
      drillLow: 'Rhythm drill: as the front foot lifts, push the hands back toward the catcher, then swing.', drillHigh: 'Keep the knob over your back foot during the load.',
      fix: (f, i) => { const tgt = clamp(loadD, 2, 14); const dx = -(tgt - loadD)/100; const w = bump(i, ST, L, L, LA); shiftJ(f, [9,10], dx, 0, w); shiftJ(f, [7,8], dx, 0, w*0.6); } });
    const kneeC = angAt(P(at(C), S.lHip), P(at(C), S.lKn), P(at(C), S.lAn));
    add({ key: 'leadKnee', label: 'Front knee at contact', value: kneeC, unit: '°', lo: 152, hi: 180, tol: 14, weight: 1.1, phase: 'Contact',
      low: 'Front leg stays soft at contact. A firm front leg stops your forward move and whips the barrel.',
      drillLow: 'Front-leg brace drill: from a wide stance, push the front heel into the ground as you swing.',
      fix: (f, i) => { const w = bump(i, FP, C - sec(0.03), C + sec(0.1), FI); kneeTo(f, S, 'l', Math.max(160, kneeC), w, true); } });
    const kneeB = angAt(P(at(C), S.bHip), P(at(C), S.bKn), P(at(C), S.bAn));
    add({ key: 'backKnee', label: 'Back knee at contact', value: kneeB, unit: '°', lo: 95, hi: 150, tol: 15, weight: 0.5, phase: 'Contact',
      low: 'Back knee collapses deep. You are sinking instead of rotating.', high: 'Back leg stays straight. You are not driving off the back side.',
      drillLow: 'Stay tall through the turn: shoot the back knee toward the front knee, not down.', drillHigh: 'Squish-the-bug drill: turn on the ball of the back foot and drive the back knee forward.',
      fix: (f, i) => { const w = bump(i, FP, C, C + sec(0.05), FI); const k = clamp(kneeB, 105, 145); const tgtAng = k; const cur = kneeB; if (Math.abs(cur - tgtAng) < 1) return; const K = P(f, S.bKn), A = P(f, S.bAn), Hh = P(f, S.bHip); const sgn = Math.sign(signedAng([Hh[0]-K[0], Hh[1]-K[1]], [A[0]-K[0], A[1]-K[1]])) || 1; rotJ(f, [S.bKn], Hh[0], Hh[1], -sgn*(tgtAng-cur)*0.35*w); } });
    const tilt = tiltBack(at(C));
    add({ key: 'tilt', label: 'Spine tilt at contact', value: tilt, unit: '°', lo: 8, hi: 35, tol: 10, weight: 1, phase: 'Contact',
      low: 'Shoulders get out in front of your hips. You are lunging at the ball.', high: 'Leaning back too far. This lifts the barrel and causes pop-ups.',
      drillLow: 'Keep the back shoulder lower than the front through the swing; stay behind the ball.', drillHigh: 'High-tee drill: swing level at a ball set at the top of the zone.',
      fix: (f, i) => { const w = bump(i, FP, C - sec(0.05), C + sec(0.12), FI); const tgt = clamp(tilt, 12, 30); const h = hipMid(f); rotJ(f, [0,1,2,3,4,5,6,7,8,9,10], h[0], h[1], (tgt - tilt)*w); } });
    const drift = (hipMid(at(C))[0] - hipMid(at(L))[0])*100;
    add({ key: 'drift', label: 'Hip shift toward pitcher', value: drift, unit: '% ht', lo: 2, hi: 16, tol: 6, weight: 0.8, phase: 'Contact',
      low: 'Weight stays on the back side. You are spinning without moving into the ball.', high: 'Hips slide too far forward. Your weight is out over the front foot.',
      drillLow: 'Walk-through drill: take a step back, then stride and swing to feel the forward move.', drillHigh: 'Hold a wall against your front hip: rotate into it, do not slide past it.',
      fix: (f, i) => { const tgt = clamp(drift, 2, 16); const dx = (tgt - drift)/100; const w = bump(i, L, C, FI, n+1); shiftJ(f, [...HEAD, 5,6,7,8,9,10,11,12], dx, 0, w); shiftJ(f, [S.lKn, S.bKn], dx, 0, w*0.5); } });
    const ttc = (C - LA)/fps;
    add({ key: 'ttc', label: 'Launch to contact time', value: ttc*1000, unit: 'ms', lo: 120, hi: 200, tol: 50, weight: 0.6, phase: 'Launch',
      low: 'Very quick launch. Check that this was a full swing.', high: 'Long swing. The hands take a slow, wide route to the ball.',
      drillLow: '', drillHigh: 'Short-bat or knob-to-ball drill: lead with the knob and keep the barrel tight to the back shoulder.', fix: null });
    add({ key: 'seq', label: 'Kinematic sequence', value: kin.score*100, unit: '%', lo: 99, hi: 100, tol: 30, weight: 1.4, phase: 'Launch', order: kin.order,
      low: `Energy order was ${kin.order.join(' → ')}. Ideal is hips first, then torso, then hands${kin.segs.some(s=>s.name==='Bat')?', then bat':''}. When a later segment fires early, speed leaks out.`,
      drillLow: 'Hip-turn drill: bat behind the back, fire the hips toward the pitcher while the shoulders stay closed for a beat.', fix: null });
    const hm = hipMid(at(FI)); const a1 = at(FI)[S.lAn*2], a2 = at(FI)[S.bAn*2];
    const bal = (hm[0] - (Math.min(a1, a2) + Math.max(a1, a2))/2) / Math.max(0.05, Math.abs(a1 - a2)/2) * 100;
    add({ key: 'balance', label: 'Finish balance (hips over base)', value: bal, unit: '%', lo: -60, hi: 70, tol: 40, weight: 0.5, phase: 'Finish',
      low: 'Falling back at the finish.', high: 'Falling forward past the front foot at the finish.',
      drillLow: 'Hold your finish for 2 seconds after every tee swing.', drillHigh: 'Hold your finish for 2 seconds after every tee swing.',
      fix: (f, i) => { const tgt = clamp(bal, -60, 70); const half = Math.max(0.05, Math.abs(a1-a2)/2); const dx = (tgt - bal)/100*half; const w = bump(i, C, FI - sec(0.15), n, n+1); shiftJ(f, [...HEAD, 5,6,7,8,9,10,11,12], dx, 0, w); } });

    /* ---- head: level, not dropping, eyes staying on the ball ---- */
    const drop = (nC[1] - nL[1])*100; // + = head went down
    add({ key: 'headDrop', label: 'Head height, load to contact', value: drop, unit: '% ht', lo: -3, hi: 4, tol: 4, weight: 1.2, phase: 'Contact',
      high: `Head drops ${drop.toFixed(0)}% of your height into contact. Dropping the head changes your eye level, so the ball looks like it moved.`, low: 'Head pops up into contact. Standing up pulls the barrel over the ball.',
      drillHigh: 'Stride-and-hold drill against a wall: the top of your cap should not slide down the wall when you stride and turn.', drillLow: 'Stay in your legs through the turn. Feel the back knee drive forward, not up.',
      fix: (f, i) => { const w = bump(i, L, Math.round((L+C)/2), FI, n+1); const dy = (nL[1] - P(f, 0)[1]); if (Math.abs(dy) < 0.02) return; const k = (Math.abs(dy) - 0.02)/Math.abs(dy); shiftJ(f, HEAD, 0, dy*k, w); shiftJ(f, [5,6], 0, dy*k, w*0.7); shiftJ(f, [7,8,9,10], 0, dy*k, w*0.5); } });
    const eyeTilt = f => { const a = P(f, 1), b = P(f, 2); return Math.atan2(b[1]-a[1], Math.abs(b[0]-a[0]) + 1e-4)*DEG; };
    const eyeOk = i => seq.conf ? Math.min(seq.conf[i][1], seq.conf[i][2]) > 0.25 : true;
    const tiltChange = eyeOk(L) && eyeOk(C) ? Math.abs(eyeTilt(at(C)) - eyeTilt(at(L))) : NaN;
    add({ key: 'eyes', label: 'Eyes level (tilt change, load to contact)', value: tiltChange, unit: '°', lo: 0, hi: 18, tol: 10, weight: 0.8, phase: 'Contact',
      high: 'Your eye line tilts a lot during the swing. Tilting the head changes how you see the pitch.', drillHigh: 'Soft-toss focusing on keeping both eyes level, like reading a line of text, until contact.', fix: null });
    const yaw = f => { const em = mid(f, 3, 4), ew = Math.max(0.02, Math.abs(f[6] - f[8])); return (f[0] - em[0])/ew; };
    const earOk = i => seq.conf ? Math.min(seq.conf[i][3], seq.conf[i][4]) > 0.2 : true;
    const headTurn = earOk(FP) && earOk(C) ? Math.abs(yaw(at(C)) - yaw(at(FP)))*100 : NaN;
    add({ key: 'headTurn', label: 'Head stays on the ball (turn, plant to contact)', value: headTurn, unit: '%', lo: 0, hi: 35, tol: 25, weight: 0.9, phase: 'Contact',
      high: 'Your head turns away before contact. When the head pulls off, the eyes leave the ball and the front shoulder follows.', drillHigh: 'Tee drill: after contact keep looking at the spot where the ball was for one full second.', fix: null });

    /* ---- barrel and hands ---- */
    const midL = Math.round(LA + (C - LA)*0.5);
    const tips = seq.batTips;
    const batAng = i => { const t = tips && tips[i]; if (!t || t.c < 0.2) return NaN; const h = mid(at(i), 9, 10); return Math.atan2(t.y - h[1], Math.abs(t.x - h[0]) + 1e-4)*DEG; }; // + = barrel below the hands
    let dump = NaN; if (tips && seq.batOk) { const v = []; for (let i = LA; i <= midL; i++) { const a = batAng(i); if (Number.isFinite(a)) v.push(a); } if (v.length) dump = Math.max(...v); }
    add({ key: 'dump', label: 'Barrel above the hands early in the swing', value: dump, unit: '°', lo: -90, hi: 15, tol: 15, weight: 1.2, phase: 'Launch',
      high: `The barrel drops ${Math.round(dump)}° below your hands in the first half of the swing. That's dumping the barrel: it gets long and under the ball, causing pop-ups and swing-and-miss up in the zone.`,
      drillHigh: 'High-tee drill: set the tee at the top of the zone and hit line drives. Keep the knob pointed at the ball and the barrel above the hands until they pass the back hip.', fix: null });
    const handsOut = i => dist(mid(at(i), 9, 10), P(at(i), S.bSh));
    const cast = (handsOut(midL) - handsOut(LA))*100;
    add({ key: 'cast', label: 'Hands stay close (casting)', value: cast, unit: '% ht', lo: -10, hi: 7, tol: 5, weight: 1, phase: 'Launch',
      high: 'Hands drift away from the body early. That is casting: a long, slow swing that loses the inside pitch.', drillHigh: 'Fence drill: stand a bat-length from a fence and swing without touching it. Lead with the knob, keep the back elbow tight.', fix: (f, i) => { const w = bump(i, LA, midL, midL, C); const sh = P(f, S.bSh), h = mid(f, 9, 10); const tgt = clamp(cast, -10, 6)/100; const k = (handsOut(midL) - handsOut(LA) - tgt)/Math.max(0.05, dist(h, sh)); if (k <= 0) return; shiftJ(f, [9, 10], (sh[0]-h[0])*k, (sh[1]-h[1])*k, w); shiftJ(f, [7, 8], (sh[0]-h[0])*k*0.5, (sh[1]-h[1])*k*0.5, w); } });

    /* ---- rotation: not opening early, not over-rotating ---- */
    const torsoFP = kin.torsoAngle[FP], torsoC = kin.torsoAngle[C];
    add({ key: 'earlyOpen', label: 'Shoulders closed at foot plant', value: torsoFP, unit: '°', lo: -60, hi: 30, tol: 15, weight: 1, phase: 'Foot plant',
      high: 'Your shoulders are already turning when the front foot lands. Opening early drags the barrel and takes away the outside pitch.', drillHigh: 'Stride drill: land with your front shoulder pointed at the pitcher and hold for a beat before you swing.', fix: null });
    add({ key: 'overRot', label: 'Shoulder turn at contact', value: torsoC, unit: '°', lo: 55, hi: 120, tol: 20, weight: 0.9, phase: 'Contact',
      high: 'Over-rotating: the shoulders spin well past square at contact, so the front side pulls off the ball.', low: 'Shoulders barely turn by contact. You are swinging with the arms instead of the body.',
      drillHigh: 'Opposite-field tee work: put the tee deep and hit the ball the other way.', drillLow: 'Hip-turn drill: bat behind the back, turn the belly button and chest to the pitcher.', fix: null });

    /* ---- feet: quiet back foot, planted front foot ---- */
    const backSlide = (at(C)[S.bAn*2] - at(ST)[S.bAn*2])*100;
    add({ key: 'backFoot', label: 'Back foot stays put', value: backSlide, unit: '% ht', lo: -4, hi: 7, tol: 4, weight: 0.7, phase: 'Contact',
      high: 'Back foot slides toward the pitcher. That leaks power forward instead of rotating it.', low: 'Back foot steps back toward the catcher.',
      drillHigh: 'Squish-the-bug drill: pivot on the ball of the back foot, it can turn but should not travel.', drillLow: 'Keep the back foot under you; turn on it, do not step off it.',
      fix: (f, i) => { const dx = (clamp(backSlide, -4, 7) - backSlide)/100; const w = bump(i, ST, Math.round((FP + C)/2), n, n+1); shiftJ(f, [S.bAn], dx, 0, w); shiftJ(f, [S.bKn], dx*0.6, 0, w); } });
    const frontMove = (dist(P(at(C), S.lAn), P(at(FP), S.lAn)))*100;
    add({ key: 'frontFoot', label: 'Front foot planted after landing', value: frontMove, unit: '% ht', lo: 0, hi: 5, tol: 4, weight: 0.7, phase: 'Contact',
      high: 'Front foot keeps moving after it lands. A front foot that slides or spins out cannot brace the swing.', drillHigh: 'Land and grip: after the stride, push the front heel and big toe into the ground and hold it there through contact.',
      fix: (f, i) => { if (i <= FP) return; const a = P(at(FP), S.lAn), b = P(f, S.lAn); const w = bump(i, FP, FP + 2, n, n+1); shiftJ(f, [S.lAn], a[0]-b[0], a[1]-b[1], w); shiftJ(f, [S.lKn], (a[0]-b[0])*0.6, (a[1]-b[1])*0.6, w); } });
    // informational
    const hs = speedSeries(F, fps, f => mid(f, 9, 10));
    seq.info = { handSpeed: Math.max(...hs.slice(LA, C+2)), batSpeed: seq.batSpeed && seq.batOk ? Math.max(...seq.batSpeed.slice(Math.max(0, C-3), C+3)) : null };
  } else {
    const K = g('Leg lift'), FP = g('Foot plant'), R = g('Release'), MER = g('Arm cocked'), FT = g('Follow-through');
    const lift = (at(K)[S.lHip*2+1] - at(K)[S.lKn*2+1])*100;
    add({ key: 'lift', label: 'Leg lift height (knee vs hip)', value: lift, unit: '% ht', lo: -4, hi: 14, tol: 6, weight: 0.6, phase: 'Leg lift',
      low: 'Low leg lift. You give up some of the potential energy you could drive down the mound.', high: 'Very high leg lift. Check that you stay balanced over the rubber.',
      drillLow: 'Balance-point drill: lift to belt height and hold 2 seconds before driving.', drillHigh: 'Balance-point drill: lift to belt height and hold 2 seconds before driving.',
      fix: (f, i) => { const dy = -(clamp(lift, -4, 14) - lift)/100; const w = bump(i, g('Set'), K, K, FP); shiftJ(f, [S.lKn, S.lAn], 0, dy, w); } });
    const stride = (at(FP)[S.lAn*2] - at(FP)[S.bAn*2])*100;
    add({ key: 'stride', label: 'Stride length at foot plant', value: stride, unit: '% ht', lo: 75, hi: 95, tol: 10, weight: 1.2, phase: 'Foot plant',
      low: 'Short stride. You land early and release the ball farther from the plate.', high: 'Very long stride. Your hips sink and the arm has to catch up.',
      drillLow: 'Stride-line drill: mark 80% of your height from the rubber and land on it.', drillHigh: 'Stride-line drill: mark 85% of your height and land on it, not past it.',
      fix: (f, i) => { const dx = (clamp(stride, 75, 95) - stride)/100; const w = bump(i, K, FP, n, n+1); shiftJ(f, [S.lAn], dx, 0, w); shiftJ(f, [S.lKn], dx, 0, w*0.8); shiftJ(f, [...HEAD, 5,6,7,8,9,10,11,12], dx, 0, w*0.45); shiftJ(f, [S.bKn], dx, 0, w*0.3); } });
    const kFP = angAt(P(at(FP), S.lHip), P(at(FP), S.lKn), P(at(FP), S.lAn));
    const kR = angAt(P(at(R), S.lHip), P(at(R), S.lKn), P(at(R), S.lAn));
    add({ key: 'kneeFP', label: 'Front knee at foot plant', value: kFP, unit: '°', lo: 125, hi: 160, tol: 14, weight: 0.7, phase: 'Foot plant',
      low: 'Landing in a deep bend. You will struggle to brace the front leg.', high: 'Landing stiff-legged. This puts stress on the knee and tips the head forward early.',
      drillLow: 'Land softly but with the knee over the ankle, then push back into the ground.', drillHigh: 'Land with a slight bend, like landing from a small hop.',
      fix: (f, i) => { const w = bump(i, K, FP, FP, R); kneeTo(f, S, 'l', clamp(kFP, 130, 155), w, true); } });
    const block = kR - kFP;
    add({ key: 'block', label: 'Front leg brace (plant to release)', value: block, unit: '°', lo: 0, hi: 40, tol: 12, weight: 1.1, phase: 'Release',
      low: `Front knee bends another ${Math.abs(block).toFixed(0)}° after landing. The leg is giving way, so energy leaks instead of transferring up to the arm.`, high: 'Front leg snaps very straight. Fine if it is smooth and you are not jarring the knee.',
      drillLow: 'Rocker drill: land, then push the front heel down and straighten the front leg as you throw.', drillHigh: '',
      fix: (f, i) => { const w = bump(i, FP, R - 1, R + Math.round(fps*0.1), FT); kneeTo(f, S, 'l', Math.max(kFP + 8, kR), w, true); } });
    const elbowH = (at(FP)[S.bSh*2+1] - at(FP)[S.bEl*2+1])*100;
    add({ key: 'elbow', label: 'Throwing elbow height at foot plant', value: elbowH, unit: '% ht', lo: -5, hi: 6, tol: 5, weight: 1, phase: 'Foot plant',
      low: 'Throwing elbow is below the shoulder at landing. This loads the shoulder and arm.', high: 'Elbow very high above the shoulder at landing.',
      drillLow: 'Mirror drill: at your landing position, check the elbow is level with the shoulder line.', drillHigh: 'Mirror drill: at your landing position, check the elbow is level with the shoulder line.',
      fix: (f, i) => { const w = bump(i, K, FP, MER, R); const sh = P(f, S.bSh), el = P(f, S.bEl); const tgtY = sh[1] - clamp(elbowH, -3, 5)/100; const L = dist(sh, el); const dy = tgtY - sh[1]; const dxs = Math.sign(el[0]-sh[0]) || -1; const nx = Math.sqrt(Math.max(0, L*L - dy*dy)) * dxs; const deg = signedAng([el[0]-sh[0], el[1]-sh[1]], [nx, dy]); rotJ(f, [S.bEl, S.bWr], sh[0], sh[1], deg*w); } });
    const armUp = (at(FP)[S.bEl*2+1] - at(FP)[S.bWr*2+1])*100;
    add({ key: 'armTiming', label: 'Arm up at foot plant (hand over elbow)', value: armUp, unit: '% ht', lo: 2, hi: 30, tol: 6, weight: 1, phase: 'Foot plant',
      low: 'Throwing hand is below the elbow when the front foot lands. The arm is late and has to rush to catch up.', high: '',
      drillLow: 'Rhythm drill: break hands earlier so the ball is up, facing second base, when the front foot lands.', drillHigh: '',
      fix: (f, i) => { const w = bump(i, K, FP, MER, R); const el = P(f, S.bEl), wr = P(f, S.bWr); const L = dist(el, wr); const tgt = [el[0] - L*0.35, el[1] - L*0.94]; const deg = signedAng([wr[0]-el[0], wr[1]-el[1]], [tgt[0]-el[0], tgt[1]-el[1]]); rotJ(f, [S.bWr], el[0], el[1], deg*w); } });
    const hm = mid(at(R), 11, 12), sm = mid(at(R), 5, 6); const fTilt = Math.atan2(sm[0]-hm[0], -(sm[1]-hm[1]))*DEG;
    add({ key: 'trunk', label: 'Trunk forward tilt at release', value: fTilt, unit: '°', lo: 25, hi: 50, tol: 10, weight: 1, phase: 'Release',
      low: 'Upright at release. You are not getting out over your front leg.', high: 'Bent far over at release. Your head and eyes drop off the target.',
      drillLow: 'Towel drill: reach the towel out to a target 6 feet in front of your stride foot.', drillHigh: 'Keep your eyes level on the catcher\'s mitt through release.',
      fix: (f, i) => { const w = bump(i, FP, R - 1, R + 2, FT); const h = mid(f, 11, 12); rotJ(f, [0,1,2,3,4,5,6,7,8,9,10], h[0], h[1], (clamp(fTilt, 28, 46) - fTilt)*w); } });
    const tuck = dist(P(at(R), S.lWr), mid(at(R), 5, 6))*100;
    add({ key: 'glove', label: 'Glove arm tucked at release', value: tuck, unit: '% ht', lo: 0, hi: 20, tol: 8, weight: 0.7, phase: 'Release',
      low: '', high: 'Glove flies away from the body. It pulls the front shoulder open early.',
      drillLow: '', drillHigh: 'Bring your chest to the glove, keep the glove out front and over the front knee.',
      fix: (f, i) => { const w = bump(i, FP, R - 1, R + 3, FT); const s = mid(f, 5, 6); const g2 = P(f, S.lWr); const target = [s[0] + 0.06, s[1] + 0.06]; shiftJ(f, [S.lWr], (target[0]-g2[0]), (target[1]-g2[1]), w); const e = P(f, S.lEl); shiftJ(f, [S.lEl], (target[0]-e[0])*0.5 + 0.0, (target[1]+0.04-e[1])*0.5, w); } });
    const ext = (at(R)[S.bWr*2] - at(FP)[S.lAn*2])*100;
    add({ key: 'release', label: 'Release point vs front foot', value: ext, unit: '% ht', lo: -12, hi: 20, tol: 8, weight: 0.7, phase: 'Release',
      low: 'Releasing well behind the front foot. You lose extension and the hitter sees the ball longer.', high: '',
      drillLow: 'Long-toss with a focus on finishing the arm out over the front knee.', drillHigh: '',
      fix: (f, i) => { const dx = (clamp(ext, -12, 20) - ext)/100; const w = bump(i, MER, R, R + 1, FT); shiftJ(f, [S.bWr], dx, 0, w); shiftJ(f, [S.bEl], dx*0.6, 0, w); shiftJ(f, [...HEAD, 5, 6], dx*0.35, 0, w); } });
    add({ key: 'seq', label: 'Kinematic sequence', value: kin.score*100, unit: '%', lo: 99, hi: 100, tol: 30, weight: 1.4, phase: 'Foot plant', order: kin.order,
      low: `Energy order was ${kin.order.join(' → ')}. Ideal is hips, then torso, then elbow, then hand. When the arm fires early it does the work the legs should.`,
      drillLow: 'Hip-lead drill: from the stride position, rotate the hips toward the plate while the chest stays closed for a beat.', fix: null });
    const slot = Math.atan2(-(at(R)[S.bWr*2+1] - at(R)[S.bSh*2+1]), at(R)[S.bWr*2] - at(R)[S.bSh*2])*DEG;
    const ws = speedSeries(F, fps, f => P(f, S.bWr));
    seq.info = { armSlot: slot, handSpeed: ws[R] };
  }
  metrics.forEach(m => { m.status = m.score == null ? 'na' : m.score >= 85 ? 'good' : m.score >= 60 ? 'watch' : 'fix'; const show = m.status !== 'good'; m.cue = !show ? '' : m.value < m.lo ? m.low : m.value > m.hi ? m.high : ''; m.drill = !show ? '' : m.value < m.lo ? m.drillLow : m.value > m.hi ? m.drillHigh : ''; });
  const scored = metrics.filter(m => m.score != null);
  const tw = scored.reduce((s, m) => s + m.weight, 0) || 1;
  const overall = Math.round(scored.reduce((s, m) => s + m.score*m.weight, 0)/tw);
  // ideal ghost frames
  const fixes = metrics.filter(m => m.fix && m.score != null && m.score < 85);
  const ideal = F.map((f, i) => { const o = Float32Array.from(f); for (const m of fixes) m.fix(o, i); return o; });
  return { kind, phases, kin, metrics, overall, ideal, fixes: fixes.map(m => m.key) };
}

/* ---------- storage format + comparison ---------- */
function packRecord(seq, res, meta) {
  const a = res.phases.start, b = res.phases.end;
  const fr = []; for (let i = a; i <= b; i++) fr.push(Array.from(seq.frames[i], v => Math.round(v*1000)));
  const bat = seq.batTips ? seq.batTips.slice(a, b+1).map(t => t && t.c > 0.2 ? [Math.round(t.x*1000), Math.round(t.y*1000)] : 0) : null;
  return { v: 1, kind: res.kind, fps: seq.fps, leadLeft: seq.leadLeft, frames: fr, bat,
    phases: res.phases.list.map(([nm, i]) => [nm, clamp(i - a, 0, b - a)]), key: res.phases.key - a,
    metrics: Object.fromEntries(res.metrics.map(m => [m.key, Number.isFinite(m.value) ? Math.round(m.value*10)/10 : null])),
    order: res.kin.order, overall: res.overall, info: seq.info || {}, cues: res.metrics.filter(m => m.status === 'fix' || m.status === 'watch').sort((x,y)=>x.score-y.score).slice(0,3).map(m => ({ key: m.key, label: m.label, cue: m.cue })),
    ...meta };
}
function unpackFrames(rec) { return rec.frames.map(a => Float32Array.from(a, v => v/1000)); }
function seqFromRecord(rec) {
  const frames = unpackFrames(rec);
  const seq = { frames, conf: frames.map(() => new Float32Array(NJ).fill(1)), fps: rec.fps, leadLeft: rec.leadLeft, H: 1, groundY: 0, originX: 0, fwd: 1,
    batTips: rec.bat ? rec.bat.map(t => t ? { x: t[0]/1000, y: t[1]/1000, c: 1 } : null) : null, stored: true };
  seq.fixedPhases = { list: rec.phases.map(p => [p[0], clamp(p[1], 0, frames.length - 1)]), start: 0, end: frames.length - 1, key: clamp(rec.key, 0, frames.length - 1), keyName: rec.kind === 'hit' ? 'Contact' : 'Release' };
  if (seq.batTips) addBatSpeed(seq);
  return seq;
}
// map a current frame index to a fractional stored index, aligning phases piecewise-linearly
function alignIndex(curPhases, recPhases, i, recLen) {
  const pts = []; for (const [nm, ci] of curPhases) { const r = recPhases.find(p => p[0] === nm); if (r) pts.push([ci, r[1]]); }
  pts.sort((a, b) => a[0] - b[0]);
  if (!pts.length) return clamp(i, 0, recLen-1);
  if (i <= pts[0][0]) return clamp(pts[0][1] - (pts[0][0] - i), 0, recLen-1);
  for (let k = 1; k < pts.length; k++) if (i <= pts[k][0]) { const [a0, b0] = pts[k-1], [a1, b1] = pts[k]; return clamp(b0 + (b1-b0)*(i-a0)/Math.max(1e-6, a1-a0), 0, recLen-1); }
  const last = pts[pts.length-1]; return clamp(last[1] + (i - last[0]), 0, recLen-1);
}
function lerpFrame(frames, fi) { const a = Math.floor(fi), b = Math.min(frames.length-1, a+1), t = fi - a; const fa = frames[a], fb = frames[b]; const o = new Float32Array(NJ*2); for (let k = 0; k < o.length; k++) o[k] = fa[k] + (fb[k]-fa[k])*t; return o; }
const DELTA_TOL = { stride: 5, head: 3, load: 3, leadKnee: 10, backKnee: 12, tilt: 7, drift: 5, ttc: 35, seq: 20, balance: 35, lift: 5, kneeFP: 10, block: 10, elbow: 4, armTiming: 5, trunk: 7, glove: 6, release: 6, headDrop: 3, eyes: 8, headTurn: 18, dump: 12, cast: 4, earlyOpen: 12, overRot: 18, backFoot: 4, frontFoot: 3 };
function compareRecords(curMetrics, rec) {
  return curMetrics.filter(m => Number.isFinite(m.value) && rec.metrics[m.key] != null).map(m => {
    const d = m.value - rec.metrics[m.key]; const tol = DELTA_TOL[m.key] || m.tol/2;
    return { key: m.key, label: m.label, unit: m.unit, now: m.value, then: rec.metrics[m.key], delta: d, off: Math.abs(d) > tol, big: Math.abs(d) > 2*tol };
  });
}

/*PRO*/
function addBatSpeed(seq) {
  if (!seq.batTips) return;
  const t = seq.batTips; const ok = t.filter(b => b && b.c > 0.2).length / t.length;
  const xs = t.map(b => b ? b.x : NaN), ys = t.map(b => b ? b.y : NaN);
  const fill = arr => { let last = NaN; const o = arr.map(v => Number.isFinite(v) ? (last = v) : last); let nx = NaN; for (let i = o.length-1; i >= 0; i--) { if (Number.isFinite(arr[i])) nx = arr[i]; if (!Number.isFinite(o[i])) o[i] = nx; } return o; };
  const fx = fill(xs), fy = fill(ys);
  const vx = deriv(fx, seq.fps), vy = deriv(fy, seq.fps);
  seq.batSpeed = gauss1d(vx.map((v, i) => Number.isFinite(v) ? Math.hypot(v, vy[i]) : 0), 0.8);
  seq.batOk = ok > 0.55;
}

if (typeof module !== 'undefined') module.exports = { NJ, JN, sideMap, cleanRaw, normalize, toPixel, solveBat, solveBat2, detectPhases, analyzeMotion, packRecord, seqFromRecord, alignIndex, lerpFrame, compareRecords, buildDemo, buildPro, buildFaulty, buildModel, MODELS, addBatSpeed, NA, angAt, P, mid, gauss1d, deriv, speedSeries };
