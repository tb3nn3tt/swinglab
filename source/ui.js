(() => {
const $ = id => document.getElementById(id);
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;' }[c]));
const fmtDate = ts => new Date(ts).toLocaleString([], { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' });

const st = {
  mode: 'hit', tab: 'Analyze', hand: { hit: 'R', pitch: 'R' }, slow: 1,
  seq: null, res: null, source: 'demo', name: 'Sample swing', hasVideo: false, times: null,
  frame: 0, fpos: 0, playing: false, speed: 0.5,
  layers: { video: true, skel: true, bat: true, energy: false, ideal: true, past: true },
  focusFix: null, idealFrames: null, compareRec: null, compareOff: 0, records: [], prefs: {}, recFrameCache: new Map(),
  heightIn: 70, busy: false, cancel: false, lastTs: 0, energy: null, saved: false,
};
try { const p = JSON.parse(localStorage.getItem('swinglab.ui') || '{}'); if (p.hand) st.hand = p.hand; if (p.mode) st.mode = p.mode; if (p.heightIn) st.heightIn = p.heightIn; } catch {}
const saveUi = () => { try { localStorage.setItem('swinglab.ui', JSON.stringify({ hand: st.hand, mode: st.mode, heightIn: st.heightIn })); } catch {} };

/* ---------------- storage: claude db (per-viewer, private) with local fallback ---------------- */
const store = {
  kind: 'local', db: null, col: null, unsub: null,
  readLocal() { try { return JSON.parse(localStorage.getItem('swinglab.v1') || '{"records":[],"prefs":{}}'); } catch { return { records: [], prefs: {} }; } },
  writeLocal(o) { try { localStorage.setItem('swinglab.v1', JSON.stringify(o)); return true; } catch { return false; } },
  init() {
    const l = this.readLocal(); st.records = l.records || []; st.prefs = l.prefs || {}; renderAll();
    const use = window.claude && window.claude.use;
    if (!use) return;
    (async () => {
      try {
        const [db, user] = await Promise.all([window.claude.use('db'), window.claude.use('user')]);
        const uid = user && await user.id();
        if (!db || !uid) return;
        this.db = db; this.col = db.collection('data/users/' + uid); this.kind = 'cloud';
        $('storeTag').textContent = 'Synced to your account';
        // move anything saved on this device into the account
        const local = this.readLocal();
        for (const r of local.records || []) { try { await this.col.doc(r.id).set(r); } catch {} }
        if (local.prefs && Object.keys(local.prefs).length) { try { await this.col.doc('prefs').set({ type: 'prefs', ...local.prefs }); } catch {} }
        if ((local.records || []).length) this.writeLocal({ records: [], prefs: {} });
        const timer = setTimeout(() => toast('Saved reps are taking a while to load.'), 10000);
        this.unsub = this.col.onSnapshot(snap => {
          if (!snap.metadata.fromCache) clearTimeout(timer);
          const recs = [], prefs = {};
          for (const d of snap.docs) { const v = d.data(); if (!v) continue; if (d.id === 'prefs') Object.assign(prefs, v); else if (v.kind) recs.push({ ...v, id: d.id }); }
          st.records = recs; st.prefs = prefs; renderAll();
        }, () => { clearTimeout(timer); });
      } catch {}
    })();
  },
  async save(rec) {
    if (this.kind === 'cloud') { await this.col.doc(rec.id).set(rec); return; }
    const l = this.readLocal(); l.records = (l.records || []).filter(r => r.id !== rec.id).concat(rec);
    if (!this.writeLocal(l)) throw new Error('This device ran out of space for saved reps.');
    st.records = l.records; renderAll();
  },
  async remove(id) {
    if (this.kind === 'cloud') { await this.col.doc(id).delete(); return; }
    const l = this.readLocal(); l.records = (l.records || []).filter(r => r.id !== id); this.writeLocal(l); st.records = l.records; renderAll();
  },
  async setPrefs(p) {
    st.prefs = { ...st.prefs, ...p };
    if (this.kind === 'cloud') { const { type, ...rest } = st.prefs; await this.col.doc('prefs').set({ type: 'prefs', ...rest }); return; }
    const l = this.readLocal(); l.prefs = st.prefs; this.writeLocal(l); renderAll();
  },
};
const recsFor = kind => st.records.filter(r => r.kind === kind).sort((a, b) => b.createdAt - a.createdAt);
const baselineFor = kind => { const id = st.prefs['base_' + kind]; return st.records.find(r => r.id === id && r.kind === kind) || null; };

/* ---------------- copy per mode ---------------- */
const MODE = {
  hit: { title: 'Film a swing', rec: 'Film or upload a swing', hand: 'Bats', key: 'Contact', noun: 'swing', nouns: 'swings', demoName: 'Sample swing',
    tips: ['Stand 15–20 ft away on the far side of the plate, facing the hitter\'s chest.', 'Hold the phone at waist height and keep it still. A fence or tripod helps.', 'Keep the whole body and the full bat in frame from load to finish.', 'Start recording 2 seconds before the swing. One swing per clip.', 'Use slo-mo if your phone has it, and pick the matching clip speed above.'] },
  pitch: { title: 'Film a pitch', rec: 'Film or upload a pitch', hand: 'Throws', key: 'Release', noun: 'pitch', nouns: 'pitches', demoName: 'Sample delivery',
    tips: ['Film from the side, square to the line between the mound and the plate, 20–30 ft away.', 'Hold the phone at chest height and keep it still.', 'Keep the whole body in frame from the set position through the follow-through.', 'Start recording before the leg lift. One pitch per clip.', 'Use slo-mo if your phone has it, and pick the matching clip speed above.'] },
};

/* ---------------- models ---------------- */
// fast pose model for the live camera, precise pose model for analysis, object detector to lock onto the player and find the bat
const NETS = { fast: { url: 'models/model.json', size: 192 }, precise: { url: 'models/thunder/model.json', size: 256 }, det: { url: 'models/yolo/model.json', size: 320, out: 'model/tf_detect/concat_3' } };
const nets = {}, netP = {};
let tfP = null;
function tfReady() {
  if (!window.tf) return Promise.reject(new Error('The tracking library did not load. Check your connection and reload.'));
  if (!tfP) tfP = (async () => { try { await tf.setBackend('webgl'); } catch { await tf.setBackend('cpu'); } await tf.ready(); })();
  return tfP;
}
async function loadNet(key) {
  if (nets[key]) return nets[key];
  if (!netP[key]) netP[key] = (async () => {
    await tfReady(); const spec = NETS[key]; const m = await tf.loadGraphModel(spec.url);
    if (key === 'det') tf.tidy(() => m.execute(tf.zeros([1, 320, 320, 3]), spec.out)); else tf.tidy(() => m.execute(tf.zeros([1, spec.size, spec.size, 3], 'int32')));
    return m;
  })();
  try { nets[key] = await netP[key]; } catch (e) { delete netP[key]; throw e; }
  return nets[key];
}
let model = null; // fast pose model, used by the live camera
async function ensureModel(onMsg) { onMsg && onMsg('Starting the tracker…'); model = await loadNet('fast'); return model; }
async function ensurePrecise(onMsg) { onMsg && onMsg('Loading the precise tracker…'); try { return await loadNet('precise'); } catch { return ensureModel(); } }
async function ensureDetector() { try { return await loadNet('det'); } catch { return null; } }

/* ---------------- video analysis ---------------- */
const vid = $('vid');
const cans = {};
function canFor(s) { if (!cans[s]) { const c = document.createElement('canvas'); c.width = c.height = s; cans[s] = { c, x: c.getContext('2d', { willReadFrequently: true }) }; } return cans[s]; }
// draw a square crop of the source into an s×s canvas, black outside the frame (avoids Safari's out-of-bounds drawImage bug)
function drawCrop(x, src, crop, s) {
  const W = src.videoWidth || src.width, H = src.videoHeight || src.height;
  x.fillStyle = '#000'; x.fillRect(0, 0, s, s);
  const sx0 = Math.max(0, crop.x), sy0 = Math.max(0, crop.y), sx1 = Math.min(W, crop.x + crop.s), sy1 = Math.min(H, crop.y + crop.s);
  if (sx1 <= sx0 || sy1 <= sy0) return; const k = s/crop.s;
  x.drawImage(src, sx0, sy0, sx1 - sx0, sy1 - sy0, (sx0 - crop.x)*k, (sy0 - crop.y)*k, (sx1 - sx0)*k, (sy1 - sy0)*k);
}
const gcv = document.createElement('canvas'); const gx = gcv.getContext('2d', { willReadFrequently: true });
function waitEvt(el, ev, ms = 4000) { return new Promise(res => { let t; const h = () => { clearTimeout(t); el.removeEventListener(ev, h); res(true); }; el.addEventListener(ev, h); t = setTimeout(() => { el.removeEventListener(ev, h); res(false); }, ms); }); }
async function seek(t) { if (Math.abs(vid.currentTime - t) < 1e-4) return; const p = waitEvt(vid, 'seeked', 3000); vid.currentTime = t; await p; }
async function runPose(crop, src = vid, net = model) {
  const s = net === nets.precise ? 256 : 192, { c, x } = canFor(s);
  drawCrop(x, src, crop, s);
  const out = tf.tidy(() => net.execute(tf.browser.fromPixels(c).toInt().expandDims(0)));
  const d = await out.data(); out.dispose();
  const k = new Float32Array(51); let sc = 0;
  for (let j = 0; j < 17; j++) { k[j*3] = crop.x + d[j*3+1]*crop.s; k[j*3+1] = crop.y + d[j*3]*crop.s; k[j*3+2] = d[j*3+2]; sc += d[j*3+2]; }
  return { k, score: sc/17, used: crop };
}
function cropFrom(k, W, H) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity, c = 0;
  for (let j = 0; j < 17; j++) if (k[j*3+2] > 0.2) { c++; x0 = Math.min(x0, k[j*3]); x1 = Math.max(x1, k[j*3]); y0 = Math.min(y0, k[j*3+1]); y1 = Math.max(y1, k[j*3+1]); }
  if (c < 6) return null;
  const s = Math.min(Math.max(W, H)*1.2, Math.max(0.2*Math.min(W, H), Math.max(x1-x0, y1-y0)*1.6));
  return { x: (x0+x1)/2 - s/2, y: (y0+y1)/2 - s/2, s };
}
async function poseAt(W, H, prevCrop, src = vid, net = model) {
  const full = { x: (W - Math.max(W, H))/2, y: (H - Math.max(W, H))/2, s: Math.max(W, H) };
  let r = await runPose(prevCrop || full, src, net);
  if (r.score < 0.25 && prevCrop) r = await runPose(full, src, net);
  const c = cropFrom(r.k, W, H);
  if (!prevCrop && c) { const r2 = await runPose(c, src, net); if (r2.score >= r.score*0.9) r = r2; }
  return { ...r, crop: r.score > 0.25 ? cropFrom(r.k, W, H) : null };
}
/* object detector: people (to lock on) and baseball bats */
function iou(a, b) { const x0 = Math.max(a.x - a.w/2, b.x - b.w/2), x1 = Math.min(a.x + a.w/2, b.x + b.w/2), y0 = Math.max(a.y - a.h/2, b.y - b.h/2), y1 = Math.min(a.y + a.h/2, b.y + b.h/2); const i = Math.max(0, x1 - x0)*Math.max(0, y1 - y0); return i/(a.w*a.h + b.w*b.h - i + 1e-6); }
function nms(list, th) { list.sort((p, q) => q.c - p.c); const out = []; for (const b of list) { if (out.every(o => iou(o, b) < th)) out.push(b); if (out.length >= 8) break; } return out; }
async function detectAt(src, crop) {
  const net = nets.det; if (!net) return null;
  const { c, x } = canFor(320); drawCrop(x, src, crop, 320);
  const t = tf.tidy(() => net.execute(tf.browser.fromPixels(c).toFloat().div(255).expandDims(0), NETS.det.out));
  const d = await t.data(); t.dispose();
  const people = [], bats = [], n = d.length/85;
  for (let i = 0; i < n; i++) {
    const o = i*85, ob = d[o+4]; if (ob < 0.05) continue;
    const pc = ob*d[o+5], bc = ob*d[o+39]; if (pc < 0.25 && bc < 0.08) continue;
    const b = { x: crop.x + d[o]*crop.s, y: crop.y + d[o+1]*crop.s, w: d[o+2]*crop.s, h: d[o+3]*crop.s };
    if (pc >= 0.25) people.push({ ...b, c: pc }); if (bc >= 0.08) bats.push({ ...b, c: bc });
  }
  return { people: nms(people, 0.5), bats: nms(bats, 0.4) };
}
const bodyBox = k => { let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity, c = 0; for (let j = 0; j < 17; j++) if (k[j*3+2] > 0.3) { c++; x0 = Math.min(x0, k[j*3]); x1 = Math.max(x1, k[j*3]); y0 = Math.min(y0, k[j*3+1]); y1 = Math.max(y1, k[j*3+1]); } return c >= 6 ? { x: (x0+x1)/2, y: (y0+y1)/2, w: x1 - x0, h: y1 - y0 } : null; };
// one tracked frame: pose in the crop that follows the player; if the pose jumps or fades, re-find the same player with the detector
async function trackFrame(W, H, tr, net) {
  let r = await poseAt(W, H, tr.crop, vid, net);
  const bb = bodyBox(r.k), ref = tr.box;
  const jumped = bb && ref && Math.hypot(bb.x - ref.x, bb.y - ref.y) > 0.45*Math.max(ref.h, ref.w);
  if ((r.score < 0.32 || !bb || jumped) && nets.det) {
    const side = ref ? Math.max(ref.h, ref.w)*3 : Math.max(W, H);
    const search = ref ? { x: ref.x - side/2, y: ref.y - side/2, s: side } : { x: (W - side)/2, y: (H - side)/2, s: side };
    const dd = await detectAt(vid, search);
    if (dd && dd.people.length) {
      const pick = ref ? dd.people.slice().sort((p, q) => Math.hypot(p.x - ref.x, p.y - ref.y) - Math.hypot(q.x - ref.x, q.y - ref.y))[0] : dd.people.slice().sort((p, q) => q.w*q.h - p.w*p.h)[0];
      const s = Math.max(pick.w, pick.h)*1.3, r2 = await runPose({ x: pick.x - s/2, y: pick.y - s/2, s }, vid, net);
      const b2 = bodyBox(r2.k);
      if (r2.score > r.score*0.9 || jumped) { r = { ...r2, crop: r2.score > 0.25 ? cropFrom(r2.k, W, H) : null }; }
      if (b2) tr.box = b2;
    }
  }
  const b = bodyBox(r.k);
  if (b && r.score > 0.3) tr.box = tr.box ? { x: tr.box.x*0.4 + b.x*0.6, y: tr.box.y*0.4 + b.y*0.6, w: tr.box.w*0.6 + b.w*0.4, h: tr.box.h*0.6 + b.h*0.4 } : b;
  // smooth the crop so it does not jitter frame to frame
  if (r.crop) tr.crop = tr.crop ? { x: tr.crop.x*0.5 + r.crop.x*0.5, y: tr.crop.y*0.5 + r.crop.y*0.5, s: tr.crop.s*0.6 + r.crop.s*0.4 } : r.crop;
  return r;
}
/* bat: brightness edges + motion along rays from the hands, boosted where the detector sees a bat */
const batState = { prev: null, lens: [] };
function batScores(k, W, H, dets) {
  const GW = 256, sc = GW / W, GH = Math.round(H*sc);
  if (gcv.width !== GW || gcv.height !== GH) { gcv.width = GW; gcv.height = GH; batState.prev = null; }
  gx.drawImage(vid, 0, 0, GW, GH);
  const im = gx.getImageData(0, 0, GW, GH).data;
  const gray = new Float32Array(GW*GH); for (let i = 0; i < gray.length; i++) gray[i] = 0.3*im[i*4] + 0.59*im[i*4+1] + 0.11*im[i*4+2];
  const prev = batState.prev; batState.prev = gray;
  const I = (x, y) => { const xi = x|0, yi = y|0; if (xi < 0 || yi < 0 || xi >= GW || yi >= GH) return -1; return gray[yi*GW + xi]; };
  const M = (x, y) => { if (!prev) return 0; const xi = x|0, yi = y|0; if (xi < 0 || yi < 0 || xi >= GW || yi >= GH) return 0; return Math.abs(gray[yi*GW + xi] - prev[yi*GW + xi]); };
  const hx = (k[27] + k[30])/2*sc, hy = (k[28] + k[31])/2*sc;
  const Hg = Math.max(20, (((k[45+1] + k[48+1])/2) - k[1]) / 0.88 * sc);
  const out = new Float32Array(NA); const r0 = 0.07*Hg, r1 = 0.5*Hg, d = Math.max(2, 0.035*Hg), steps = 22;
  const limb = [7, 8, 5, 6].map(j => Math.atan2(k[j*3+1]*sc - hy, k[j*3]*sc - hx));
  for (let a = 0; a < NA; a++) {
    const ang = a*5*Math.PI/180, cx = Math.cos(ang), cy = Math.sin(ang), nx = -cy, ny = cx; let sum = 0, mot = 0;
    for (let s = 0; s < steps; s++) {
      const r = r0 + (r1 - r0)*s/(steps-1), x = hx + cx*r, y = hy + cy*r;
      const c = I(x, y), s1 = I(x + nx*d, y + ny*d), s2 = I(x - nx*d, y - ny*d); if (c < 0 || s1 < 0 || s2 < 0) continue;
      sum += Math.max(0, Math.abs(c - (s1 + s2)/2) - Math.abs(s1 - s2)*0.5); mot += M(x, y);
    }
    let v = sum/steps + 0.35*mot/steps;
    for (const ea of limb) { let dd = Math.abs(ang - ea); dd = Math.min(dd, 2*Math.PI - dd); if (dd < 0.3) v *= 0.35; }
    out[a] = v;
  }
  // detector boost: the bat's far end is the box corner farthest from the hands
  let mx = 0; for (const v of out) mx = Math.max(mx, v);
  const hpx = (k[27] + k[30])/2, hpy = (k[28] + k[31])/2, bodyPx = Hg/sc;
  for (const b of (dets && dets.bats) || []) {
    const inside = hpx > b.x - b.w/2 - 0.12*bodyPx && hpx < b.x + b.w/2 + 0.12*bodyPx && hpy > b.y - b.h/2 - 0.12*bodyPx && hpy < b.y + b.h/2 + 0.12*bodyPx;
    if (!inside) continue;
    let best = null; for (const [cx2, cy2] of [[b.x - b.w/2, b.y - b.h/2], [b.x + b.w/2, b.y - b.h/2], [b.x - b.w/2, b.y + b.h/2], [b.x + b.w/2, b.y + b.h/2]]) { const dd = Math.hypot(cx2 - hpx, cy2 - hpy); if (!best || dd > best.d) best = { d: dd, x: cx2, y: cy2 }; }
    if (best.d < 0.2*bodyPx) continue;
    batState.lens.push(best.d/bodyPx);
    const ta = Math.atan2(best.y - hpy, best.x - hpx);
    for (let a = 0; a < NA; a++) { let dd = Math.abs(a*5*Math.PI/180 - ta); dd = Math.min(dd, 2*Math.PI - dd); out[a] += (mx || 1)*1.6*b.c*Math.exp(-(dd*dd)/(2*0.2*0.2)); }
  }
  return out;
}

async function analyzeFile(file, opts = {}) {
  if (st.busy) return;
  st.busy = true; st.cancel = false; pause();
  const kind = st.mode, leadLeft = kind === 'hit' ? st.hand.hit === 'R' : st.hand.pitch === 'R', slow = opts.slow || st.slow;
  const proc = $('proc'), bar = $('procBar'), txt = $('procTxt');
  const msg = (t, p) => { txt.textContent = t; if (p != null) bar.style.width = Math.round(p*100) + '%'; };
  proc.hidden = false; msg('Loading video…', 0);
  if (st.videoUrl) URL.revokeObjectURL(st.videoUrl);
  st.videoUrl = URL.createObjectURL(file);
  st.seq = null; st.res = null; st.hasVideo = true; st.source = 'video'; st.compareRec = null; st.saved = false;
  st.name = currentPlayer().name + ' · ' + MODE[kind].noun + ' · ' + fmtDate(Date.now()); st.viewHeight = null;
  setStageMeta(); renderReport();
  try {
    vid.src = st.videoUrl; vid.muted = true; vid.playsInline = true;
    const ok = await waitEvt(vid, 'loadeddata', 15000);
    if (!ok || !vid.videoWidth) throw new Error('This video could not be opened here. Try a different clip, or record with the camera button.');
    try { await vid.play(); vid.pause(); } catch {}
    if (!Number.isFinite(vid.duration) || vid.duration <= 0) { // clips recorded in the browser often report no length until seeked to the end
      const p = waitEvt(vid, 'durationchange', 3000); try { vid.currentTime = 1e7; } catch {} await p; await seek(0);
    }
    const vdur = Number.isFinite(vid.duration) && vid.duration > 0 ? vid.duration : (opts.duration || 0);
    if (!vdur) throw new Error('This video has no readable length. Try recording again or upload it from your library.');
    sizeStage();
    const net = await ensurePrecise(t => msg(t, 0.02));
    msg(kind === 'hit' ? 'Loading the player and bat finder…' : 'Loading the player finder…', 0.03); await ensureDetector();
    const W = vid.videoWidth, H = vid.videoHeight, dur = Math.min(vdur, 30*slow);
    const realDur = dur/slow;
    let tA = 0, tB = dur;
    // lock on: the biggest person in the first frame is the player
    const tr = { crop: null, box: null };
    await seek(Math.min(dur*0.1, 0.2*slow));
    if (nets.det) { const side = Math.max(W, H), dd = await detectAt(vid, { x: (W - side)/2, y: (H - side)/2, s: side }); if (dd && dd.people.length) { const p = dd.people.slice().sort((a, b) => b.w*b.h - a.w*a.h)[0]; tr.box = { x: p.x, y: p.y, w: p.w, h: p.h }; const s = Math.max(p.w, p.h)*1.3; tr.crop = { x: p.x - s/2, y: p.y - s/2, s }; } }
    if (realDur > 3.6) {
      // coarse pass: find the moment of fastest hand movement
      const step = slow/6, ts = [], ks = []; const ctr = { ...tr };
      for (let t = 0; t < dur; t += step) {
        if (st.cancel) throw new Error('cancel');
        await seek(t); const r = await trackFrame(W, H, ctr, model || net); ts.push(t); ks.push(r.k);
        drawLive(r.k, W, H); msg('Finding the ' + MODE[kind].noun + '…', 0.05 + 0.2*t/dur);
      }
      const sp = ks.map((k, i) => { if (!i) return 0; const p = ks[i-1]; const v = j => k[j*3+2] > 0.2 && p[j*3+2] > 0.2 ? Math.hypot(k[j*3]-p[j*3], k[j*3+1]-p[j*3+1]) : 0; return Math.max(v(9), v(10)); });
      const pk = ts[sp.indexOf(Math.max(...sp))] || dur/2;
      const pre = (kind === 'hit' ? 1.9 : 2.4)*slow, post = (kind === 'hit' ? 1.0 : 1.1)*slow;
      tA = Math.max(0, pk - pre); tB = Math.min(dur, pk + post);
    }
    const perReal = slow > 1 ? 60 : 30, mstep = slow/perReal;
    const times = []; for (let t = tA; t <= tB + 1e-6; t += mstep) times.push(t);
    const raw = [], dets = []; let found = 0; batState.prev = null; batState.lens = [];
    for (let i = 0; i < times.length; i++) {
      if (st.cancel) throw new Error('cancel');
      await seek(times[i]);
      const r = await trackFrame(W, H, tr, net); raw.push(r.k); if (r.score > 0.3) found++;
      if (kind === 'hit' && nets.det && r.score > 0.25) {
        const hx = (r.k[27] + r.k[30])/2, hy = (r.k[28] + r.k[31])/2, bh = tr.box ? Math.max(tr.box.h, tr.box.w) : Math.max(W, H)*0.5, s = bh*1.35;
        dets.push(await detectAt(vid, { x: hx - s/2, y: hy - s/2, s }));
      } else dets.push(null);
      drawLive(r.k, W, H);
      msg('Tracking joints' + (kind === 'hit' ? ' and bat' : '') + '… frame ' + (i+1) + ' of ' + times.length, 0.3 + 0.62*(i+1)/times.length);
    }
    if (found < times.length*0.4) throw new Error('Could not find a player in most frames. Keep the whole body in frame, in good light, with nobody else close by.');
    msg('Scoring mechanics…', 0.95);
    const clean = cleanRaw(raw, 1.0);
    const nz = normalize(clean, leadLeft);
    const seq = { ...nz, fps: perReal, slow, vw: W, vh: H, kind }; // H stays the player's body height in pixels
    if (kind === 'hit') {
      // second look at each frame for the bat, now that the joints are cleaned up
      const scores = [];
      batState.prev = null;
      for (let i = 0; i < times.length; i++) {
        if (st.cancel) throw new Error('cancel');
        await seek(times[i]);
        scores.push(batScores(clean[i], W, H, dets[i]));
        msg('Following the bat… frame ' + (i+1) + ' of ' + times.length, 0.95 + 0.04*(i+1)/times.length);
      }
      const hs = clean.map((k, i) => { const p = clean[Math.max(0, i-1)]; return Math.hypot((k[27]+k[30])/2 - (p[27]+p[30])/2, (k[28]+k[31])/2 - (p[28]+p[31])/2)/nz.H; });
      const allow = hs.map(v => 2 + 80*v*(30/perReal)*slow);
      const path = solveBat(scores, allow);
      const lens = batState.lens.slice().sort((a, b) => a - b);
      const L = (lens.length >= 3 ? clamp(lens[lens.length >> 1], 0.32, 0.62) : 0.45)*nz.H;
      seq.batTips = clean.map((k, i) => { const p = path[i]; if (!scores[i]) return null; const hx = (k[27] + k[30])/2, hy = (k[28] + k[31])/2; const a = p.angleIdx*5*Math.PI/180; const tx = hx + Math.cos(a)*L, ty = hy + Math.sin(a)*L; return { x: (tx - nz.originX)*nz.fwd/nz.H, y: (ty - nz.groundY)/nz.H, c: p.conf }; });
      addBatSpeed(seq);
    }
    st.times = times; st.seq = seq;
    st.res = analyzeMotion(seq, kind);
    afterAnalysis();
    if (nz.autoSwapped) toast('Lead leg looked like the other side, so the tracker used the leg that strides. Check the Bats / Throws setting.');
  } catch (e) {
    if (e.message === 'cancel') { loadDemo(); toast('Analysis canceled.'); }
    else { loadDemo(); toast(e.message || 'Something went wrong while analyzing.'); }
  } finally { proc.hidden = true; st.busy = false; }
}
function drawLive(k, W, H) {
  const { ctx, w, h } = canvasCtx(); ctx.clearRect(0, 0, w, h);
  const fit = videoFit(W, H, w, h); const f = new Float32Array(34); for (let j = 0; j < 17; j++) { f[j*2] = k[j*3]; f[j*2+1] = k[j*3+1]; }
  drawSkel(ctx, f, (x, y) => [fit.x + x*fit.s, fit.y + y*fit.s], { color: css('--lead'), scale: fit.s*(H*0.9), dim: k });
}

/* ---------------- demo + library sources ---------------- */
function loadDemo() {
  pause();
  const seq = buildDemo(st.mode);
  st.seq = seq; st.res = analyzeMotion(seq, st.mode); st.source = 'demo'; st.hasVideo = false; st.times = null; st.name = MODE[st.mode].demoName; st.saved = false; st.viewHeight = null; st.recId = null;
  vid.removeAttribute('src'); vid.load();
  st.compareRec = null;
  afterAnalysis(true);
}
function loadRecord(rec) {
  pause();
  const seq = seqFromRecord(rec);
  st.seq = seq; st.res = analyzeMotion(seq, rec.kind); st.source = 'library'; st.hasVideo = false; st.times = null; st.name = rec.name; st.saved = true; st.recId = rec.id; st.viewHeight = rec.heightIn || null;
  vid.removeAttribute('src'); vid.load();
  st.compareRec = null;
  afterAnalysis(true);
}
function afterAnalysis(quiet) {
  const r = st.res;
  st.focusFix = null; st.idealFrames = r.ideal;
  computeEnergy();
  st.frame = r.phases.start; st.fpos = st.frame;
  if (!st.compareRec) { const b = baselineFor(st.mode); if (b && b.id !== st.recId && st.source !== 'demo') st.compareRec = b; }
  prepCompare();
  setStageMeta(); sizeStage(); buildScrub(); renderReport(); seekFrame(r.phases.key);
  if (!quiet) { setTab('Analyze'); toast('Analysis ready. Score ' + r.overall + '.'); }
}
function computeEnergy() {
  const F = st.seq.frames, fps = st.seq.fps, n = F.length;
  const sp = F.map((f, i) => { const a = F[Math.max(0, i-1)], b = F[Math.min(n-1, i+1)]; const dt = (Math.min(n-1, i+1) - Math.max(0, i-1))/fps || 1; const o = new Float32Array(17); for (let j = 0; j < 17; j++) o[j] = Math.hypot(b[j*2]-a[j*2], b[j*2+1]-a[j*2+1])/dt; return o; });
  const all = []; sp.forEach(o => o.forEach(v => all.push(v))); all.sort((a, b) => a - b);
  const ref = all[Math.floor(all.length*0.97)] || 1;
  st.energy = sp.map(o => o.map(v => Math.min(1, v/ref)));
}
function prepCompare() {
  const rec = st.compareRec; if (!rec || rec.kind !== st.mode) { st.compareRec = null; return; }
  if (!st.recFrameCache.has(rec.id)) st.recFrameCache.set(rec.id, unpackFrames(rec));
  const rf = st.recFrameCache.get(rec.id); const S1 = sideMap(st.seq.leadLeft), S2 = sideMap(rec.leadLeft);
  const cur = st.seq.frames[st.res.phases.start];
  st.compareOff = cur[S1.bAn*2] - rf[0][S2.bAn*2];
}
function pastFrameAt(i) {
  const rec = st.compareRec; if (!rec) return null;
  const rf = st.recFrameCache.get(rec.id); const fi = alignIndex(st.res.phases.list, rec.phases, i, rf.length);
  const f = lerpFrame(rf, fi); for (let j = 0; j < 17; j++) f[j*2] += st.compareOff; return f;
}

/* ---------------- drawing ---------------- */
const cv = $('cv');
function canvasCtx(canvas = cv) {
  const dpr = Math.min(2, window.devicePixelRatio || 1); const r = canvas.getBoundingClientRect();
  const W = Math.max(1, Math.round(r.width*dpr)), H = Math.max(1, Math.round(r.height*dpr));
  if (canvas.width !== W || canvas.height !== H) { canvas.width = W; canvas.height = H; }
  const ctx = canvas.getContext('2d'); return { ctx, w: W, h: H, dpr };
}
function videoFit(vw, vh, w, h) { const s = Math.min(w/vw, h/vh); return { s, x: (w - vw*s)/2, y: (h - vh*s)/2 }; }
const BONES = [[5,7,'l'],[7,9,'l'],[6,8,'b'],[8,10,'b'],[5,6,'c'],[5,11,'c'],[6,12,'c'],[11,12,'c'],[11,13,'l'],[13,15,'l'],[12,14,'b'],[14,16,'b']];
function heat(t) {
  const stops = [[0, [43,76,126]], [0.35, [63,167,245]], [0.6, [232,180,74]], [0.85, [240,101,79]], [1, [255,241,214]]];
  for (let i = 1; i < stops.length; i++) if (t <= stops[i][0]) { const [a, ca] = stops[i-1], [b, cb] = stops[i]; const u = (t - a)/(b - a); return `rgb(${ca.map((v, k) => Math.round(v + (cb[k]-v)*u)).join(',')})`; }
  return 'rgb(255,241,214)';
}
// opts: {color, colors:{l,b,c}, dash, alpha, scale (px per body height), energy (Float32Array 17), leadLeft, dim}
function drawSkel(ctx, f, map, o) {
  const S = sideMap(o.leadLeft !== false);
  const sideOf = j => { if (j <= 4) return 'c'; const isLeft = j % 2 === 1; return isLeft === (o.leadLeft !== false) ? 'l' : 'b'; };
  const lw = Math.max(2, o.scale*0.012);
  ctx.save(); ctx.globalAlpha = o.alpha ?? 1; ctx.lineCap = 'round'; ctx.lineJoin = 'round';
  if (o.dash) ctx.setLineDash([lw*2.2, lw*1.6]);
  const pt = j => map(f[j*2], f[j*2+1]);
  const colOf = (a, b, side) => o.energy ? heat((o.energy[a] + o.energy[b])/2) : o.color || (o.colors ? o.colors[side] : '#fff');
  for (const [a, b, tag] of BONES) {
    if (o.dim && (o.dim[a*3+2] < 0.15 || o.dim[b*3+2] < 0.15)) continue;
    const side = tag === 'c' ? 'c' : sideOf(b);
    const p = pt(a), q = pt(b); ctx.strokeStyle = colOf(a, b, side); ctx.lineWidth = side === 'c' ? lw*0.85 : lw*1.15;
    ctx.beginPath(); ctx.moveTo(p[0], p[1]); ctx.lineTo(q[0], q[1]); ctx.stroke();
  }
  // neck + head
  const sm = map((f[10]+f[12])/2, (f[11]+f[13])/2), nz = pt(0);
  ctx.strokeStyle = o.energy ? heat(o.energy[0]) : o.color || (o.colors ? o.colors.c : '#fff'); ctx.lineWidth = lw*0.85;
  ctx.beginPath(); ctx.moveTo(sm[0], sm[1]); ctx.lineTo(nz[0] + (sm[0]-nz[0])*0.35, nz[1] + (sm[1]-nz[1])*0.35); ctx.stroke();
  ctx.beginPath(); ctx.arc(nz[0], nz[1], o.scale*0.055, 0, Math.PI*2); ctx.stroke();
  // joints
  ctx.setLineDash([]);
  for (const j of [5,6,7,8,9,10,11,12,13,14,15,16]) {
    if (o.dim && o.dim[j*3+2] < 0.15) continue;
    const p = pt(j); const r = Math.max(2.5, o.scale*(j === 9 || j === 10 ? 0.014 : 0.011));
    ctx.beginPath(); ctx.arc(p[0], p[1], r, 0, Math.PI*2);
    if (o.dash) { ctx.fillStyle = o.color; ctx.globalAlpha = (o.alpha ?? 1)*0.9; ctx.fill(); }
    else { ctx.fillStyle = o.energy ? heat(o.energy[j]) : o.color || (o.colors ? o.colors[sideOf(j)] : '#fff'); ctx.fill(); ctx.lineWidth = Math.max(1, lw*0.35); ctx.strokeStyle = 'rgba(7,12,18,.8)'; ctx.stroke(); }
  }
  ctx.restore();
}
function arrow(ctx, a, b, col, w) {
  const dx = b[0]-a[0], dy = b[1]-a[1], L = Math.hypot(dx, dy); if (L < 6) return;
  const ux = dx/L, uy = dy/L, hs = Math.min(10, L*0.4) + w;
  ctx.save(); ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = w; ctx.lineCap = 'round';
  ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0] - ux*hs*0.6, b[1] - uy*hs*0.6); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(b[0], b[1]); ctx.lineTo(b[0] - ux*hs - uy*hs*0.5, b[1] - uy*hs + ux*hs*0.5); ctx.lineTo(b[0] - ux*hs + uy*hs*0.5, b[1] - uy*hs - ux*hs*0.5); ctx.closePath(); ctx.fill();
  ctx.restore();
}
let viewBox = null;
function computeViewBox() {
  const r = st.res, F = st.seq.frames; let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = 0.02;
  const take = f => { for (let j = 0; j < 17; j++) { x0 = Math.min(x0, f[j*2]); x1 = Math.max(x1, f[j*2]); y0 = Math.min(y0, f[j*2+1]); y1 = Math.max(y1, f[j*2+1]); } };
  for (let i = r.phases.start; i <= r.phases.end; i++) { take(F[i]); take(r.ideal[i]); const b = st.seq.batTips && st.seq.batTips[i]; if (b && b.c > 0.2) { x0 = Math.min(x0, b.x); x1 = Math.max(x1, b.x); y0 = Math.min(y0, b.y); y1 = Math.max(y1, b.y); } }
  const pad = 0.12; viewBox = { x0: x0 - pad, x1: x1 + pad, y0: y0 - pad, y1: y1 + pad*0.6 };
}
function stageMap(w, h) {
  if (st.hasVideo) { const fit = videoFit(st.seq.vw, st.seq.vh, w, h); return { map: (x, y) => { const p = toPixel(st.seq, x, y); return [fit.x + p[0]*fit.s, fit.y + p[1]*fit.s]; }, scale: st.seq.H*fit.s }; }
  const vb = viewBox; const s = Math.min(w/(vb.x1 - vb.x0), h/(vb.y1 - vb.y0)); const ox = (w - (vb.x1 - vb.x0)*s)/2 - vb.x0*s, oy = (h - (vb.y1 - vb.y0)*s)/2 - vb.y0*s;
  return { map: (x, y) => [ox + x*s, oy + y*s], scale: s };
}
function draw() {
  const { ctx, w, h, dpr } = canvasCtx(); ctx.clearRect(0, 0, w, h);
  if (!st.seq || !st.res) return;
  const i = Math.round(st.frame), r = st.res, f = st.seq.frames[i];
  const { map, scale } = stageMap(w, h);
  if (st.hasVideo && st.layers.video) { ctx.fillStyle = 'rgba(7,12,18,.32)'; ctx.fillRect(0, 0, w, h); }
  if (!st.hasVideo || !st.layers.video) {
    // field: ground line + chalk grid in body heights
    ctx.save(); ctx.strokeStyle = 'rgba(233,238,244,.07)'; ctx.lineWidth = 1*dpr;
    const g0 = map(-3, 0), step = scale*0.25;
    for (let x = ((g0[0] % step) + step) % step; x < w; x += step) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke(); }
    for (let y = g0[1]; y > 0; y -= step) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }
    ctx.strokeStyle = 'rgba(233,238,244,.35)'; ctx.lineWidth = 2*dpr; ctx.beginPath(); ctx.moveTo(0, g0[1]); ctx.lineTo(w, g0[1]); ctx.stroke();
    ctx.fillStyle = 'rgba(233,238,244,.45)'; ctx.font = `${11*dpr}px ${css('--f-mono') || 'monospace'}`; ctx.fillText(st.mode === 'hit' ? 'PITCHER →' : 'PLATE →', w - 86*dpr, g0[1] + 16*dpr);
    ctx.restore();
  }
  const colors = { l: css('--lead'), b: css('--back'), c: '#E9EEF4' };
  // past rep
  if (st.layers.past && st.compareRec) { const pf = pastFrameAt(i); if (pf) drawSkel(ctx, pf, map, { color: css('--past'), scale, alpha: 0.85, leadLeft: st.compareRec.leadLeft }); }
  // fixed form ghost + arrows
  const ideal = st.idealFrames;
  if (st.layers.ideal && ideal && st.res.fixes.length) {
    const g = ideal[i]; const gcol = css('--ghost');
    drawSkel(ctx, g, map, { color: gcol, dash: true, scale, alpha: 0.95, leadLeft: st.seq.leadLeft });
    for (let j = 0; j < 17; j++) { if (j > 0 && j < 5) continue; const d = Math.hypot(g[j*2]-f[j*2], g[j*2+1]-f[j*2+1]); if (d > 0.025) arrow(ctx, map(f[j*2], f[j*2+1]), map(g[j*2], g[j*2+1]), gcol, Math.max(1.5, scale*0.006)); }
  }
  // bat path
  const tips = st.seq.batTips;
  if (st.layers.bat && tips && st.mode === 'hit') {
    ctx.save(); ctx.lineCap = 'round';
    const a = Math.max(r.phases.start, i - 14);
    for (let k = a + 1; k <= i; k++) { const p = tips[k-1], q = tips[k]; if (!p || !q || p.c < 0.2 || q.c < 0.2) continue; const t = (k - a)/(i - a + 1); ctx.strokeStyle = `rgba(232,180,74,${0.15 + 0.7*t})`; ctx.lineWidth = Math.max(2, scale*0.01)*(0.5 + t); const P1 = map(p.x, p.y), Q1 = map(q.x, q.y); ctx.beginPath(); ctx.moveTo(P1[0], P1[1]); ctx.lineTo(Q1[0], Q1[1]); ctx.stroke(); }
    const b = tips[i]; if (b && b.c > 0.2) { const hnd = map((f[18]+f[20])/2, (f[19]+f[21])/2), tp = map(b.x, b.y); ctx.strokeStyle = '#D9A55B'; ctx.lineWidth = Math.max(3, scale*0.022); ctx.beginPath(); ctx.moveTo(hnd[0], hnd[1]); ctx.lineTo(tp[0], tp[1]); ctx.stroke(); ctx.strokeStyle = 'rgba(7,12,18,.7)'; ctx.lineWidth = Math.max(1, scale*0.004); ctx.stroke(); }
    ctx.restore();
  }
  if (st.layers.skel) drawSkel(ctx, f, map, { colors, scale, leadLeft: st.seq.leadLeft, energy: st.layers.energy ? st.energy[i] : null });
  // hud
  const ph = currentPhase(i); $('hudPhase').textContent = ph;
  const key = r.phases.key; $('hudTime').textContent = ((i - key)/st.seq.fps*1000 >= 0 ? '+' : '') + Math.round((i - key)/st.seq.fps*1000) + ' ms';
  $('fNo').textContent = (i - r.phases.start + 1) + ' / ' + (r.phases.end - r.phases.start + 1);
  const sc = $('scrub'); if (+sc.value !== i) sc.value = i;
  document.querySelectorAll('#phaseChips .chip').forEach(c => c.setAttribute('aria-current', c.dataset.name === ph ? 'true' : 'false'));
  drawChart();
}
function currentPhase(i) { let nm = st.res.phases.list[0][0]; for (const [n, k] of st.res.phases.list) if (i >= k) nm = n; return nm; }

/* ---------------- kinematic sequence chart ---------------- */
function drawChart() {
  const el = $('chartCv'); if (!el || !st.res) return;
  const { ctx, w, h, dpr } = canvasCtx(el); ctx.clearRect(0, 0, w, h);
  const r = st.res, a = r.phases.start, b = r.phases.end, fps = st.seq.fps;
  const padL = 8*dpr, padR = 8*dpr, padT = 14*dpr, padB = 22*dpr;
  const X = i => padL + (i - a)/(b - a)*(w - padL - padR), Y = v => padT + (1 - v)*(h - padT - padB);
  ctx.strokeStyle = css('--line'); ctx.lineWidth = 1*dpr; ctx.beginPath(); ctx.moveTo(padL, Y(0)); ctx.lineTo(w - padR, Y(0)); ctx.stroke();
  ctx.fillStyle = css('--ink-2'); ctx.font = `${10.5*dpr}px ${css('--f-mono') || 'monospace'}`; ctx.textAlign = 'center';
  const key = r.phases.key;
  for (let ms = -1000; ms <= 1000; ms += 100) { const i = key + ms/1000*fps; if (i < a || i > b) continue; const x = X(i); ctx.fillRect(x - 0.5*dpr, Y(0), 1*dpr, 4*dpr); if (ms % 200 === 0) ctx.fillText(ms === 0 ? MODE[st.mode].key : (ms > 0 ? '+' : '') + ms, x, h - 6*dpr); }
  ctx.setLineDash([4*dpr, 4*dpr]); ctx.strokeStyle = css('--ink-2'); ctx.beginPath(); ctx.moveTo(X(key), padT - 6*dpr); ctx.lineTo(X(key), Y(0)); ctx.stroke(); ctx.setLineDash([]);
  const cols = segColors();
  r.kin.segs.forEach((s, si) => {
    const mx = Math.max(1e-6, ...s.series.slice(a, b + 1)); ctx.strokeStyle = cols[si]; ctx.lineWidth = 2*dpr; ctx.beginPath();
    for (let i = a; i <= b; i++) { const x = X(i), y = Y(s.series[i]/mx); i === a ? ctx.moveTo(x, y) : ctx.lineTo(x, y); } ctx.stroke();
    const px = X(s.peak), py = Y(s.series[s.peak]/mx); ctx.fillStyle = cols[si]; ctx.beginPath(); ctx.arc(px, py, 4.5*dpr, 0, Math.PI*2); ctx.fill(); ctx.strokeStyle = css('--surface'); ctx.lineWidth = 2*dpr; ctx.stroke();
  });
  const x = X(Math.round(st.frame)); ctx.strokeStyle = css('--accent'); ctx.lineWidth = 2*dpr; ctx.beginPath(); ctx.moveTo(x, padT - 8*dpr); ctx.lineTo(x, Y(0)); ctx.stroke();
}
const segColors = () => [css('--c-hips'), css('--c-torso'), css('--c-arm'), css('--c-tip')];

/* ---------------- playback ---------------- */
function frameAtTime(t) { const T = st.times; if (!T || T.length < 2) return 0; const step = T[1] - T[0]; return Math.max(0, Math.min(T.length - 1, Math.round((t - T[0])/step))); }
function seekFrame(i) {
  const r = st.res; if (!r) return; i = Math.max(r.phases.start, Math.min(r.phases.end, Math.round(i)));
  st.frame = i; st.fpos = i;
  if (st.hasVideo && st.times) { try { vid.currentTime = st.times[i]; } catch {} }
  draw();
}
function play() {
  if (!st.res || st.busy) return; st.playing = true; setPlayIcon();
  if (st.frame >= st.res.phases.end) seekFrame(st.res.phases.start);
  if (st.hasVideo) { vid.playbackRate = st.speed; vid.play().catch(() => {}); }
  st.lastTs = performance.now();
}
function pause() { st.playing = false; setPlayIcon(); if (!vid.paused) vid.pause(); }
function setPlayIcon() { $('bPlay').setAttribute('aria-label', st.playing ? 'Pause' : 'Play'); $('icoPlay').innerHTML = st.playing ? '<path d="M6 4h4v16H6zM14 4h4v16h-4z"/>' : '<path d="M7 4v16l13-8z"/>'; }
function loop(ts) {
  if (st.playing && st.res) {
    const r = st.res;
    if (st.hasVideo) {
      const i = frameAtTime(vid.currentTime); st.frame = i;
      if (i >= r.phases.end || vid.ended) { vid.pause(); seekFrame(r.phases.start); setTimeout(() => { if (st.playing) vid.play().catch(() => {}); }, 350); }
    } else {
      const dt = (ts - st.lastTs)/1000; st.fpos += dt*st.seq.fps*st.speed;
      if (st.fpos > r.phases.end + st.seq.fps*0.4) st.fpos = r.phases.start;
      st.frame = Math.min(r.phases.end, st.fpos);
    }
    draw();
  }
  st.lastTs = ts; drawGameDay(ts); compareLoop(ts);
  requestAnimationFrame(loop);
}

/* ---------------- stage + controls ---------------- */
function sizeStage() {
  const stg = $('stage');
  if (st.hasVideo && vid.videoWidth) { stg.style.aspectRatio = vid.videoWidth + ' / ' + vid.videoHeight; }
  else stg.style.aspectRatio = '4 / 5';
  vid.style.opacity = st.hasVideo && st.layers.video ? '1' : '0';
  if (st.res) computeViewBox();
}
function setStageMeta() {
  $('srcName').textContent = st.name;
  $('srcTag').textContent = st.source === 'demo' ? 'Sample data' : st.source === 'library' ? 'From library' : 'Your video';
  document.querySelector('[data-layer="video"]').hidden = !st.hasVideo;
  document.querySelector('[data-layer="bat"]').hidden = st.mode !== 'hit';
  document.querySelector('[data-layer="past"]').hidden = !st.compareRec;
  const lg = [`<span><i style="background:var(--lead)"></i>Lead side</span>`, `<span><i style="background:var(--back)"></i>${st.mode === 'hit' ? 'Back side' : 'Throwing side'}</span>`];
  if (st.res && st.res.fixes.length) lg.push(`<span><i class="dash"></i>Fixed form</span>`);
  if (st.compareRec) lg.push(`<span><i style="background:var(--past)"></i>${esc(st.compareRec.name)}</span>`);
  if (st.layers.energy) lg.push(`<span><i style="background:linear-gradient(90deg,#2B4C7E,#3FA7F5,#E8B44A,#F0654F)"></i>Segment speed</span>`);
  $('legend').innerHTML = lg.join('');
}
function buildScrub() {
  const r = st.res, sc = $('scrub'); sc.min = r.phases.start; sc.max = r.phases.end; sc.value = st.frame;
  const span = r.phases.end - r.phases.start || 1;
  $('ticks').innerHTML = r.phases.list.map(([n, i]) => `<i class="${i === r.phases.key ? 'key' : ''}" style="left:${(i - r.phases.start)/span*100}%" title="${esc(n)}"></i>`).join('');
  $('phaseChips').innerHTML = `<span class="row-lbl" style="align-self:center;margin-right:4px">Phase</span>` + r.phases.list.map(([n]) => `<button class="chip phase" data-name="${esc(n)}">${esc(n)}</button>`).join('');
}

/* ---------------- report ---------------- */
function rangeBar(m, past) {
  const span = (m.hi - m.lo) || 1; let lo = m.lo - span*0.8, hi = m.hi + span*0.8;
  const vals = [m.value, past].filter(Number.isFinite); for (const v of vals) { lo = Math.min(lo, v - span*0.1); hi = Math.max(hi, v + span*0.1); }
  const X = v => ((v - lo)/(hi - lo)*100).toFixed(1) + '%';
  return `<div class="range" aria-hidden="true"><div class="track"></div><div class="band" style="left:${X(m.lo)};width:calc(${X(m.hi)} - ${X(m.lo)})"></div>${Number.isFinite(past) ? `<div class="mk past" style="left:${X(past)}"></div>` : ''}<div class="mk" style="left:${X(m.value)}"></div></div>`;
}
const mph = v => Number.isFinite(v) && v > 0 ? (v*(st.viewHeight || st.heightIn)*0.0254*2.23694).toFixed(0) + ' mph' : '—';
const fmtNum = (v, unit) => { const r = Math.abs(v) >= 100 || unit === 'ms' || unit === '%' || Number.isInteger(v) ? Math.round(v) : Math.round(v*10)/10; return Object.is(r, -0) ? 0 : r; };
const fmtVal = (v, unit) => !Number.isFinite(v) ? '—' : fmtNum(v, unit) + (unit === '°' ? '°' : unit === '% ht' ? '% of height' : ' ' + unit);
const fmtShort = (v, unit) => !Number.isFinite(v) ? '—' : fmtNum(v, unit) + (unit === '°' ? '°' : unit === '% ht' ? '%' : unit === '%' ? '%' : ' ' + unit);
const showVal = m => m.key === 'seq' ? Math.round(m.value) + '% in order' : m.key === 'balance' ? (m.value < m.lo ? 'Falling back' : m.value > m.hi ? 'Falling forward' : 'Centered') : fmtVal(m.value, m.unit);
function gradeWord(s) { return s >= 88 ? 'Game ready' : s >= 75 ? 'Solid, with fixes' : s >= 60 ? 'Needs work' : 'Rebuild the basics'; }
function renderReport() {
  const el = $('report');
  if (!st.res) { el.innerHTML = `<div class="card"><div class="eyebrow">Scouting report</div><h2>Analyzing…</h2><p class="muted">The report appears here when tracking finishes.</p></div>`; return; }
  const r = st.res, kind = r.kind, M = MODE[kind];
  const fixes = r.metrics.filter(m => m.status === 'fix' || m.status === 'watch').sort((a, b) => a.score - b.score);
  const past = st.compareRec ? st.compareRec.metrics : null;
  const cols = segColors();
  const segIdx = Object.fromEntries(r.kin.segs.map((s, i) => [s.name, i]));
  const info = st.seq.info || {};
  const recOpts = recsFor(kind).filter(x => x.id !== st.recId);
  const cmp = st.compareRec ? compareRecords(r.metrics, st.compareRec) : null;
  const offs = cmp ? cmp.filter(c => c.off) : [];
  el.innerHTML = `
  <div class="card">
    <div class="score-card">
      <div class="scoreboard">${r.overall}<small>OF 100</small></div>
      <div class="score-txt">
        <div class="eyebrow">Mechanics score · ${esc(kind === 'hit' ? 'Hitting' : 'Pitching')}</div>
        <h2>${gradeWord(r.overall)}</h2>
        <p class="muted small">${fixes.length ? `${fixes.length} thing${fixes.length > 1 ? 's' : ''} to work on. Biggest: <b style="color:var(--ink)">${esc(fixes[0].label.toLowerCase())}</b>.` : 'Every check is inside the target range.'}</p>
      </div>
    </div>
    ${st.source === 'demo' ? `<p class="note" style="margin-top:12px">This is a built-in sample ${M.noun} so you can see how the report works. Record or choose a video above to analyze your own.</p>` : ''}
  </div>

  ${cmp ? `<div class="card">
    <div class="eyebrow">Compared with</div>
    <h2>${esc(st.compareRec.name)}</h2>
    <p class="small ${offs.length ? '' : 'muted'}" style="margin-bottom:10px">${offs.length ? `${offs.length} check${offs.length > 1 ? 's are' : ' is'} out of line with that rep: ${offs.slice(0, 3).map(o => esc(o.label.toLowerCase())).join(', ')}.` : `Nothing is out of line with that rep. Your ${M.noun} matches.`}</p>
    <div class="tbl"><table class="delta"><thead><tr><th>Check</th><th style="text-align:right">Then</th><th style="text-align:right">Now</th><th style="text-align:right">Change</th></tr></thead><tbody>
    ${cmp.map(c => `<tr class="${c.big ? 'big' : c.off ? 'off' : ''}"><td>${esc(c.label)}</td><td class="num">${fmtShort(c.then, c.unit)}</td><td class="num">${fmtShort(c.now, c.unit)}</td><td class="num">${fmtNum(c.delta, c.unit) > 0 ? '+' : ''}${fmtShort(c.delta, c.unit)}</td></tr>`).join('')}
    </tbody></table></div>
    <p class="note" style="margin-top:8px">The purple skeleton on the video is that rep, lined up phase by phase. Orange and red rows changed more than normal rep-to-rep variation. Distances are percent of your height.</p>
  </div>` : ''}

  <div class="card">
    <div class="eyebrow">Energy flow</div>
    <h2>Kinematic sequence</h2>
    <p class="muted small">Speed of each body segment through the ${M.noun}. Power transfers best when each segment peaks after the one before it, from the ground up.</p>
    <div class="chart-wrap" style="margin-top:10px"><canvas id="chartCv" aria-label="Segment speed over time"></canvas></div>
    <div class="order">${r.kin.order.map((n, k) => `${k ? '<span class="muted">→</span>' : ''}<b style="background:${cols[segIdx[n]]}">${esc(n)}</b>`).join('')}<span class="pill ${r.kin.inOrder ? 'good' : 'fix'}" style="margin-left:6px">${r.kin.inOrder ? 'In order' : 'Out of order'}</span></div>
    <div class="kv" style="margin-top:12px">
      ${kind === 'hit' ? `<div><b>${mph(info.handSpeed)}</b><span>Peak hand speed</span></div>
      <div><b>${mph(info.batSpeed)}</b><span>Peak barrel speed${info.batSpeed ? ', estimate' : ''}</span></div>` :
      `<div><b>${Number.isFinite(info.armSlot) ? Math.round(info.armSlot) + '°' : '—'}</b><span>Arm slot at release (0° sidearm, 90° over the top)</span></div>
      <div><b>${mph(info.handSpeed)}</b><span>Hand speed at release, estimate</span></div>`}
    </div>
    <p class="note" style="margin-top:8px">Tap <b>Energy flow</b> under the video to color each body part by its speed and watch the wave move up the body.</p>
  </div>

  <div class="card">
    <div class="eyebrow">Checkpoints</div>
    <h2>What to adjust</h2>
    <div class="metrics">
      ${r.metrics.slice().sort((a, b) => (a.score ?? 101) - (b.score ?? 101)).map(m => `
      <div class="metric">
        <div class="m-head"><h3>${esc(m.label)}</h3><span class="pill ${m.status}">${m.status === 'good' ? 'On target' : m.status === 'watch' ? 'Watch' : m.status === 'fix' ? 'Fix' : 'No read'}</span></div>
        <div class="val">${showVal(m)}</div>
        ${m.key !== 'seq' && m.key !== 'balance' ? rangeBar(m, past ? past[m.key] : null) : ''}
        <div class="m-body">
          ${m.key !== 'seq' && m.key !== 'balance' ? `<span class="muted small">Target ${fmtVal(m.lo, m.unit)} to ${fmtVal(m.hi, m.unit)} · at ${esc(m.phase.toLowerCase())}</span>` : ''}
          ${m.cue ? `<p>${esc(m.cue)}</p>` : ''}
          ${m.drill ? `<p class="drill"><b>Drill:</b> ${esc(m.drill)}</p>` : ''}
          <div class="m-actions">
            <button class="linkbtn" data-goto="${esc(m.phase)}">Go to ${esc(m.phase.toLowerCase())}</button>
            ${r.fixes.includes(m.key) ? `<button class="linkbtn" data-focus="${m.key}">${st.focusFix === m.key ? 'Show all fixes' : 'Show only this fix'}</button>` : ''}
          </div>
        </div>
      </div>`).join('')}
    </div>
    <p class="note" style="margin-top:6px">The dashed green skeleton is your own motion with these fixes applied. Arrows show where each joint should move.</p>
  </div>

  <div class="card">
    <div class="eyebrow">Library</div>
    <h2>${st.source === 'library' ? 'Saved rep' : 'Save this rep'}</h2>
    ${st.source === 'library' ? `<p class="muted small">This rep is in your library.</p><div style="margin-top:10px"><button class="btn" data-cmpa="${esc(st.recId || '')}">Compare side by side</button></div>` : st.source === 'demo' ? `<p class="muted small">Analyze your own ${M.noun} to save it. Saved reps keep the skeleton and numbers so you can compare future reps against them.</p>` : st.saved ? `<p class="small">Saved to your library under ${esc(currentPlayer().name)}.</p><div style="margin-top:10px"><button class="btn" data-cmpa="${esc(st.recId || '')}">Compare with another player</button></div>` : `
    <div class="save-row"><input type="text" id="saveName" value="${esc(st.name)}" aria-label="Name for this rep" maxlength="60"><button class="btn primary" id="saveBtn">Save</button></div>
    <label class="check" style="margin-top:10px"><input type="checkbox" id="saveBase" ${baselineFor(kind) ? '' : 'checked'}> Make this my baseline ${M.noun}</label>
    <p class="note" style="margin-top:6px">Saves the skeleton, phases and numbers. The video stays on your phone.${store.kind === 'local' ? ' Saved on this device.' : ''}</p>`}
    <div class="opt" style="margin-top:14px"><label for="cmpSel">Compare with</label>
      <select id="cmpSel" style="flex:1;min-width:0"><option value="">Nothing</option>${recOpts.map(x => `<option value="${esc(x.id)}" ${st.compareRec && st.compareRec.id === x.id ? 'selected' : ''}>${esc(x.name)}${st.prefs['base_' + kind] === x.id ? ' ★' : ''} · ${x.overall}</option>`).join('')}</select>
    </div>
    ${!recOpts.length ? `<p class="note" style="margin-top:6px">Save a few ${M.nouns} to compare against them here.</p>` : ''}
  </div>
  <p class="note">Measured from one camera in 2D, so angles and distances are estimates. ${kind === 'hit' ? 'Bat position is estimated from the image. ' : ''}Use the numbers to compare your own reps filmed the same way.</p>`;
  requestAnimationFrame(drawChart);
}

/* ---------------- library + game day ---------------- */
function thumb(canvas, rec) {
  const frames = st.recFrameCache.get(rec.id) || unpackFrames(rec); st.recFrameCache.set(rec.id, frames);
  const f = frames[rec.key] || frames[0];
  const { ctx, w, h } = canvasCtx(canvas); ctx.clearRect(0, 0, w, h);
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity; for (let j = 0; j < 17; j++) { x0 = Math.min(x0, f[j*2]); x1 = Math.max(x1, f[j*2]); y0 = Math.min(y0, f[j*2+1]); }
  const s = Math.min(w/(x1 - x0 + 0.3), h/(0.0 - y0 + 0.25)); const ox = w/2 - (x0 + x1)/2*s, oy = h - 0.1*s;
  drawSkel(ctx, f, (x, y) => [ox + x*s, oy + y*s], { colors: { l: css('--lead'), b: css('--back'), c: '#E9EEF4' }, scale: s, leadLeft: rec.leadLeft });
}
let delArm = null;
function renderLibrary() {
  const kind = st.mode, M = MODE[kind], all = recsFor(kind), base = st.prefs['base_' + kind];
  const pnames = [...new Set(all.map(r => r.player).filter(Boolean))]; if (st.libPlayer && !pnames.includes(st.libPlayer)) st.libPlayer = null;
  const list = st.libPlayer ? all.filter(r => r.player === st.libPlayer) : all;
  $('libFilter').innerHTML = pnames.length > 1 ? `<button class="chip" data-libp="" aria-pressed="${!st.libPlayer}">Everyone</button>` + pnames.map(p => `<button class="chip" data-libp="${esc(p)}" aria-pressed="${st.libPlayer === p}">${esc(p)}</button>`).join('') : '';
  $('libEyebrow').textContent = 'Saved skeletons · ' + (kind === 'hit' ? 'Hitting' : 'Pitching');
  $('libTitle').textContent = 'Your ' + M.nouns;
  const el = $('libList');
  if (!list.length) { el.innerHTML = `<div class="empty"><p>No saved ${M.nouns} yet.</p><p class="small">Analyze a ${M.noun}, then tap Save. Your first saved ${M.noun} becomes your baseline.</p><div><button class="btn primary" data-tab="Analyze">Analyze a ${M.noun}</button></div></div>`; }
  else el.innerHTML = list.map(r => `
    <div class="lib-item">
      <canvas data-thumb="${esc(r.id)}" aria-hidden="true"></canvas>
      <div style="min-width:0"><div class="nm">${r.id === base ? '<span class="star" aria-label="Baseline">★ </span>' : ''}${esc(r.name)}</div><div class="meta">${r.player ? esc(r.player) + ' · ' : ''}${fmtDate(r.createdAt)} · ${esc(r.hand === 'L' ? 'Left' : 'Right')}</div></div>
      <div class="lib-score">${r.overall}</div>
      <div class="lib-actions">
        <button class="chip" data-open="${esc(r.id)}">Review</button>
        <button class="chip" data-cmpa="${esc(r.id)}">Side by side</button>
        <button class="chip" data-cmp="${esc(r.id)}">Overlay on current</button>
        ${r.id === base ? '' : `<button class="chip" data-base="${esc(r.id)}">Make baseline</button>`}
        ${delArm === r.id ? `<button class="chip" data-delyes="${esc(r.id)}" style="color:var(--fix);border-color:var(--fix)">Delete for good</button><button class="chip" data-delno="1">Keep</button>` : `<button class="chip" data-del="${esc(r.id)}">Delete</button>`}
      </div>
    </div>`).join('');
  el.querySelectorAll('canvas[data-thumb]').forEach(c => { const r = list.find(x => x.id === c.dataset.thumb); if (r) thumb(c, r); });
  // trend
  const tc = $('trendCard');
  if (list.length < 2) { tc.innerHTML = `<div class="eyebrow">Trend</div><h2>Rep to rep</h2><p class="muted small">Save at least two ${M.nouns} to see how your numbers move over time.</p>`; return; }
  const last = list.slice(0, 8).reverse();
  const recurring = {}; list.slice(0, 6).forEach(r => (r.cues || []).forEach(c => { recurring[c.label] = (recurring[c.label] || 0) + 1; }));
  const rec = Object.entries(recurring).sort((a, b) => b[1] - a[1]).slice(0, 4);
  tc.innerHTML = `<div class="eyebrow">Trend · last ${last.length}</div><h2>Rep to rep</h2>
    <div class="chart-wrap" style="height:140px"><canvas id="trendCv" aria-label="Score by rep"></canvas></div>
    <h3 style="margin-top:14px">Keeps showing up</h3>
    ${rec.length ? `<ul class="cue-list" style="margin-top:8px">${rec.map(([l, n]) => `<li><span class="n">${n}×</span><span>${esc(l)}</span></li>`).join('')}</ul>` : '<p class="muted small">No repeat issues in recent reps.</p>'}`;
  requestAnimationFrame(() => {
    const c = $('trendCv'); if (!c) return; const { ctx, w, h, dpr } = canvasCtx(c); ctx.clearRect(0, 0, w, h);
    const pad = 22*dpr, X = i => pad + i/(last.length - 1)*(w - pad*2), Y = v => h - pad - (v - 40)/60*(h - pad*1.6);
    ctx.strokeStyle = css('--line'); ctx.lineWidth = dpr; ctx.fillStyle = css('--ink-2'); ctx.font = `${10*dpr}px ${css('--f-mono') || 'monospace'}`;
    for (const v of [50, 75, 100]) { ctx.beginPath(); ctx.moveTo(pad, Y(v)); ctx.lineTo(w - pad, Y(v)); ctx.stroke(); ctx.fillText(String(v), 0, Y(v) + 3*dpr); }
    ctx.strokeStyle = css('--accent'); ctx.lineWidth = 2.5*dpr; ctx.beginPath(); last.forEach((r, i) => { const x = X(i), y = Y(Math.max(40, r.overall)); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.stroke();
    last.forEach((r, i) => { ctx.fillStyle = r.id === base ? css('--accent') : css('--surface'); ctx.strokeStyle = css('--accent'); ctx.lineWidth = 2*dpr; ctx.beginPath(); ctx.arc(X(i), Y(Math.max(40, r.overall)), 4.5*dpr, 0, Math.PI*2); ctx.fill(); ctx.stroke(); });
  });
}
const gd = { rec: null, frames: null, pos: 0, last: 0 };
function renderGameDay() {
  const kind = st.mode, M = MODE[kind], list = recsFor(kind);
  const base = baselineFor(kind) || list.slice().sort((a, b) => b.overall - a.overall)[0] || null;
  gd.rec = base; gd.frames = base ? (st.recFrameCache.get(base.id) || unpackFrames(base)) : null; if (base) st.recFrameCache.set(base.id, gd.frames);
  gd.demo = null;
  if (!base) { const d = buildDemo(kind); gd.demo = d; gd.frames = d.frames; gd.res = analyzeMotion(d, kind); }
  $('gdTitle').textContent = base ? (baselineFor(kind) ? 'Your baseline ' + M.noun : 'Your best ' + M.noun) : 'Sample ' + M.noun;
  $('gdNote').textContent = base ? `${base.name} · score ${base.overall}. Watch it a few times to lock in the feel before you play.` : `Save a ${M.noun} and mark it as your baseline to review it here before games.`;
  const side = $('gdSide');
  const cues = base ? (base.cues || []) : [];
  const recurring = {}; list.slice(0, 6).forEach(r => (r.cues || []).forEach(c => { recurring[c.key] = recurring[c.key] || { ...c, n: 0 }; recurring[c.key].n++; }));
  const reps = Object.values(recurring).filter(c => c.n >= 2).sort((a, b) => b.n - a.n).slice(0, 3);
  const m = base ? base.metrics : {};
  const keyNums = kind === 'hit' ? [['stride', 'Stride', '% ht'], ['head', 'Head movement', '% ht'], ['tilt', 'Spine tilt', '°'], ['ttc', 'Launch to contact', 'ms']] : [['stride', 'Stride', '% ht'], ['elbow', 'Elbow at plant', '% ht'], ['trunk', 'Trunk tilt', '°'], ['block', 'Front leg brace', '°']];
  side.innerHTML = `
    <div class="card"><div class="eyebrow">Today's focus</div><h2>Your cues</h2>
      ${cues.length || reps.length ? `<ul class="cue-list">${[...reps.map(c => ({ ...c, rep: true })), ...cues.filter(c => !reps.find(r => r.key === c.key))].slice(0, 3).map((c, i) => `<li><span class="n">${i+1}</span><div><b>${esc(c.label)}</b>${c.rep ? ` <span class="pill watch">${c.n} of last reps</span>` : ''}<p class="muted small" style="margin-top:2px">${esc(c.cue || '')}</p></div></li>`).join('')}</ul>` : `<p class="muted small">${base ? 'No open fixes on this rep. Trust it.' : 'Your cues show up here once you have saved reps.'}</p>`}
    </div>
    ${base ? `<div class="card"><div class="eyebrow">Baseline numbers</div><h2>Your baseline numbers</h2>
      <div class="kv">${keyNums.map(([k, l, u]) => `<div><b>${fmtShort(m[k], u)}</b><span>${esc(l)}${u === '% ht' ? ', % of height' : ''}</span></div>`).join('')}</div>
      <p class="note" style="margin-top:10px">Kinematic order: ${esc((base.order || []).join(' → '))}</p>
      <div style="margin-top:12px;display:flex;gap:8px;flex-wrap:wrap"><button class="btn primary" id="gdCheck">Film a rep and compare</button><button class="btn" data-open="${esc(base.id)}">Open in Analyze</button></div></div>` : ''}`;
  const b = $('gdCheck'); if (b) b.onclick = () => { st.pendingCompare = base.id; setTab('Analyze'); toast('Record or choose a ' + M.noun + '. It will be checked against ' + base.name + '.'); };
}
function drawGameDay(ts) {
  if (st.tab !== 'Game' || !gd.frames) return;
  const c = $('gdCv'); const { ctx, w, h, dpr } = canvasCtx(c); ctx.clearRect(0, 0, w, h);
  const F = gd.frames, fps = gd.rec ? gd.rec.fps : 30;
  const a = gd.rec ? 0 : gd.res.phases.start, b = gd.rec ? F.length - 1 : gd.res.phases.end;
  const dt = gd.last ? (ts - gd.last)/1000 : 0; gd.last = ts; gd.pos += dt*fps*0.5; if (gd.pos > b - a + fps*0.5) gd.pos = 0;
  const i = a + Math.min(b - a, Math.floor(gd.pos)), f = F[i];
  const tips0 = gd.rec && gd.rec.bat ? gd.rec.bat : (gd.demo && gd.demo.batTips ? gd.demo.batTips.map(t => t ? [t.x*1000, t.y*1000] : 0) : null);
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity; for (let k = a; k <= b; k++) { for (let j = 0; j < 17; j++) { x0 = Math.min(x0, F[k][j*2]); x1 = Math.max(x1, F[k][j*2]); y0 = Math.min(y0, F[k][j*2+1]); } const t = st.mode === 'hit' && tips0 && tips0[k]; if (t) { x0 = Math.min(x0, t[0]/1000); x1 = Math.max(x1, t[0]/1000); y0 = Math.min(y0, t[1]/1000); } }
  const s = Math.min(w/(x1 - x0 + 0.3), h/(-y0 + 0.3)); const ox = w/2 - (x0 + x1)/2*s, oy = h - 0.12*s;
  ctx.strokeStyle = 'rgba(233,238,244,.35)'; ctx.lineWidth = 2*dpr; ctx.beginPath(); ctx.moveTo(0, oy); ctx.lineTo(w, oy); ctx.stroke();
  const tips = gd.rec && gd.rec.bat ? gd.rec.bat : (gd.demo && gd.demo.batTips ? gd.demo.batTips.map(t => t ? [t.x*1000, t.y*1000] : 0) : null);
  if (tips && st.mode === 'hit') { const tp = tips[i]; if (tp) { const hx = (f[18]+f[20])/2, hy = (f[19]+f[21])/2; ctx.strokeStyle = '#D9A55B'; ctx.lineWidth = Math.max(3, s*0.022); ctx.lineCap = 'round'; ctx.beginPath(); ctx.moveTo(ox + hx*s, oy + hy*s); ctx.lineTo(ox + tp[0]/1000*s, oy + tp[1]/1000*s); ctx.stroke(); } }
  drawSkel(ctx, f, (x, y) => [ox + x*s, oy + y*s], { colors: { l: css('--lead'), b: css('--back'), c: '#E9EEF4' }, scale: s, leadLeft: gd.rec ? gd.rec.leadLeft : true });
  const phases = gd.rec ? gd.rec.phases : gd.res.phases.list.map(([n, k]) => [n, k - a]);
  let nm = phases[0][0]; for (const [n, k] of phases) if (i - a >= k) nm = n;
  ctx.fillStyle = '#E8B44A'; ctx.font = `600 ${13*dpr}px ${css('--f-mono') || 'monospace'}`; ctx.fillText(nm.toUpperCase(), 12*dpr, 22*dpr);
}

/* ---------------- tabs, mode, events ---------------- */
function setTab(t) {
  st.tab = t; ['Analyze', 'Compare', 'Library', 'Game'].forEach(n => { $('v' + n).hidden = n !== t; $('t' + n).setAttribute('aria-selected', n === t ? 'true' : 'false'); });
  if (t !== 'Analyze') pause();
  if (t === 'Library') renderLibrary(); if (t === 'Game') { gd.pos = 0; gd.last = 0; renderGameDay(); }
  if (t === 'Analyze') requestAnimationFrame(draw);
  if (t !== 'Compare' && cmp.playing) cmpPlay(false);
  if (t === 'Compare') renderCompare();
}
function setMode(m) {
  if (st.busy) { toast('Wait for the current analysis to finish.'); return; }
  st.mode = m; saveUi(); st.recId = null;
  $('mHit').setAttribute('aria-pressed', m === 'hit'); $('mPitch').setAttribute('aria-pressed', m === 'pitch');
  const M = MODE[m]; $('capTitle').textContent = M.title; $('recLbl').textContent = M.rec; $('handLbl').textContent = M.hand;
  $('tipsList').innerHTML = M.tips.map(t => `<li>${esc(t)}</li>`).join('');
  syncHand(); loadDemo(); renderAll();
}
function syncHand() { const h = st.hand[st.mode]; $('hR').setAttribute('aria-pressed', h === 'R'); $('hL').setAttribute('aria-pressed', h === 'L'); }
function renderAll() { syncPlayers(); if (st.tab === 'Compare' && !document.activeElement?.closest?.('#vCompare select')) renderCompare(); if (st.tab === 'Library') renderLibrary(); if (st.tab === 'Game') renderGameDay(); if (st.res && st.tab === 'Analyze' && !document.activeElement?.closest?.('#report')) renderReport(); }
let toastT; function toast(m) { const t = $('toast'); t.textContent = m; t.hidden = false; clearTimeout(toastT); toastT = setTimeout(() => t.hidden = true, 4200); }

$('mHit').onclick = () => setMode('hit'); $('mPitch').onclick = () => setMode('pitch');
$('tAnalyze').onclick = () => setTab('Analyze'); $('tLibrary').onclick = () => setTab('Library'); $('tGame').onclick = () => setTab('Game'); $('tCompare').onclick = () => setTab('Compare');
$('hR').onclick = () => { st.hand[st.mode] = 'R'; saveUi(); syncHand(); persistPlayer(); };
$('hL').onclick = () => { st.hand[st.mode] = 'L'; saveUi(); syncHand(); persistPlayer(); };
$('slowSel').onchange = e => { st.slow = +e.target.value; };
{ const hs = $('htSel'); for (let i = 48; i <= 82; i++) hs.insertAdjacentHTML('beforeend', `<option value="${i}" ${i === st.heightIn ? 'selected' : ''}>${Math.floor(i/12)}′${i%12}″</option>`); hs.onchange = e => { st.heightIn = +e.target.value; saveUi(); persistPlayer(); if (st.res) renderReport(); }; }
const onFile = e => { const f = e.target.files && e.target.files[0]; e.target.value = ''; if (!f) return; if (st.pendingCompare) { st.compareRec = st.records.find(r => r.id === st.pendingCompare) || null; st.pendingCompare = null; } analyzeFile(f).then(() => { if (st.compareRec) { prepCompare(); setStageMeta(); renderReport(); draw(); } }); };
$('upIn').onchange = onFile;
// The tap lands directly on the file input (no scripted click), which phones allow even inside embedded pages.
// If the page never loses focus after the tap, the picker was blocked: say how to get around it.
{ let pickT = 0; const left = () => { clearTimeout(pickT); };
  $('upIn').addEventListener('click', e => {
    if (st.busy) { e.preventDefault(); toast('Wait for the current analysis to finish.'); return; }
    clearTimeout(pickT); pickT = setTimeout(() => { if (document.visibilityState === 'visible' && document.hasFocus()) $('pickHelp').hidden = false; }, 1800);
  });
  window.addEventListener('blur', left); document.addEventListener('visibilitychange', left); $('upIn').addEventListener('change', () => { left(); $('pickHelp').hidden = true; });
  $('pickHelpClose').onclick = () => { $('pickHelp').hidden = true; };
}
$('procCancel').onclick = () => { st.cancel = true; };
$('bPlay').onclick = () => st.playing ? pause() : play();
$('bPrev').onclick = () => { pause(); seekFrame(st.frame - 1); };
$('bNext').onclick = () => { pause(); seekFrame(st.frame + 1); };
$('scrub').oninput = e => { pause(); seekFrame(+e.target.value); };
document.querySelectorAll('[data-speed]').forEach(b => b.onclick = () => { st.speed = +b.dataset.speed; document.querySelectorAll('[data-speed]').forEach(x => x.setAttribute('aria-pressed', x === b)); if (st.hasVideo) vid.playbackRate = st.speed; });
$('layerChips').onclick = e => { const b = e.target.closest('[data-layer]'); if (!b) return; const k = b.dataset.layer; st.layers[k] = !st.layers[k]; b.setAttribute('aria-pressed', st.layers[k]); if (k === 'video') sizeStage(); setStageMeta(); draw(); };
$('phaseChips').onclick = e => { const b = e.target.closest('[data-name]'); if (!b) return; pause(); seekFrame(st.res.phases.list.find(p => p[0] === b.dataset.name)[1]); };
vid.addEventListener('seeked', () => { if (!st.busy && !st.playing) draw(); });
document.addEventListener('keydown', e => { if (st.tab !== 'Analyze' || /INPUT|SELECT|TEXTAREA/.test(document.activeElement?.tagName)) return; if (e.key === ' ') { e.preventDefault(); st.playing ? pause() : play(); } if (e.key === 'ArrowLeft') { pause(); seekFrame(st.frame - 1); } if (e.key === 'ArrowRight') { pause(); seekFrame(st.frame + 1); } });

$('report').addEventListener('click', async e => {
  const g = e.target.closest('[data-goto]'); if (g) { pause(); seekFrame(st.res.phases.list.find(p => p[0] === g.dataset.goto)?.[1] ?? st.res.phases.key); $('stage').scrollIntoView({ behavior: 'smooth', block: 'center' }); return; }
  const fo = e.target.closest('[data-focus]'); if (fo) {
    const k = fo.dataset.focus; st.focusFix = st.focusFix === k ? null : k;
    if (st.focusFix) { const m = st.res.metrics.find(x => x.key === k); st.idealFrames = st.seq.frames.map((f, i) => { const o = Float32Array.from(f); m.fix(o, i); return o; }); pause(); seekFrame(st.res.phases.list.find(p => p[0] === m.phase)?.[1] ?? st.res.phases.key); }
    else st.idealFrames = st.res.ideal;
    st.layers.ideal = true; document.querySelector('[data-layer="ideal"]').setAttribute('aria-pressed', 'true');
    renderReport(); draw(); $('stage').scrollIntoView({ behavior: 'smooth', block: 'center' }); return;
  }
  if (e.target.id === 'saveBtn') {
    const name = ($('saveName').value || '').trim() || st.name; const asBase = $('saveBase').checked;
    const id = 'r' + Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
    const rec = packRecord(st.seq, st.res, { id, name, createdAt: Date.now(), hand: st.hand[st.mode], source: st.source, player: currentPlayer().name, playerId: currentPlayer().id, heightIn: st.heightIn });
    if (currentPlayer().unsaved) persistPlayer();
    e.target.disabled = true;
    try { await store.save(rec); if (asBase) await store.setPrefs({ ['base_' + st.mode]: id }); st.saved = true; st.recId = id; st.name = name; setStageMeta(); toast('Saved to your library' + (asBase ? ' as your baseline.' : '.')); renderReport(); }
    catch (err) { e.target.disabled = false; toast(err && err.code === 'quota_exceeded' ? 'Your library is full. Delete a few old reps and try again.' : (err && err.message) || 'Could not save. Try again.'); }
  }
});
$('report').addEventListener('change', e => {
  if (e.target.id === 'cmpSel') { st.compareRec = st.records.find(r => r.id === e.target.value) || null; prepCompare(); st.layers.past = true; document.querySelector('[data-layer="past"]').setAttribute('aria-pressed', 'true'); setStageMeta(); renderReport(); draw(); }
});
document.addEventListener('click', async e => {
  const t = e.target.closest('[data-tab]'); if (t) { setTab(t.dataset.tab); return; }
  const ca = e.target.closest('[data-cmpa]'); if (ca) { const r = st.records.find(x => x.id === ca.dataset.cmpa); if (r) { if (r.kind !== st.mode) setMode(r.kind); cmp.aId = r.id; const other = recsFor(r.kind).find(x => x.id !== r.id && (x.player || '') !== (r.player || '')) || recsFor(r.kind).find(x => x.id !== r.id); cmp.bId = other ? other.id : SAMPLE_A; if (!other) toast('Save a rep for another player to compare against. Showing a sample for now.'); setTab('Compare'); renderCompare(true); window.scrollTo({ top: 0 }); } return; }
  const lp = e.target.closest('[data-libp]'); if (lp) { st.libPlayer = lp.dataset.libp || null; renderLibrary(); return; }
  const o = e.target.closest('[data-open]'); if (o) { const r = st.records.find(x => x.id === o.dataset.open); if (r) { if (r.kind !== st.mode) setMode(r.kind); loadRecord(r); setTab('Analyze'); } return; }
  const c = e.target.closest('[data-cmp]'); if (c) { const r = st.records.find(x => x.id === c.dataset.cmp); if (r) { if (st.recId === r.id) { toast('That rep is the one open in Analyze. Open a different rep or film a new one.'); return; } st.compareRec = r; prepCompare(); st.layers.past = true; document.querySelector('[data-layer="past"]').setAttribute('aria-pressed', 'true'); setTab('Analyze'); setStageMeta(); renderReport(); draw(); } return; }
  const b = e.target.closest('[data-base]'); if (b) { try { await store.setPrefs({ ['base_' + st.mode]: b.dataset.base }); toast('Baseline updated.'); } catch { toast('Could not update the baseline. Try again.'); } return; }
  const d = e.target.closest('[data-del]'); if (d) { delArm = d.dataset.del; renderLibrary(); return; }
  const dn = e.target.closest('[data-delno]'); if (dn) { delArm = null; renderLibrary(); return; }
  const dy = e.target.closest('[data-delyes]'); if (dy) { const id = dy.dataset.delyes; delArm = null; try { await store.remove(id); if (st.prefs['base_' + st.mode] === id) await store.setPrefs({ ['base_' + st.mode]: null }); if (st.compareRec && st.compareRec.id === id) { st.compareRec = null; } toast('Deleted.'); } catch { toast('Could not delete. Try again.'); } return; }
});
new ResizeObserver(() => { if (st.res) { computeViewBox(); draw(); } }).observe($('stage'));
matchMedia('(prefers-color-scheme: dark)').addEventListener?.('change', () => draw());

/* ---------------- players ---------------- */
const players = () => (Array.isArray(st.prefs.players) ? st.prefs.players : []);
function currentPlayer() { const list = players(); return list.find(p => p.id === st.playerId) || list.find(p => p.id === st.prefs.player) || list[0] || { id: 'me', name: 'Me', heightIn: st.heightIn, hand: { ...st.hand }, unsaved: true }; }
async function savePlayer(p) { const list = players().filter(x => x.id !== p.id); list.push({ id: p.id, name: p.name, heightIn: p.heightIn, hand: { hit: p.hand.hit, pitch: p.hand.pitch } }); await store.setPrefs({ players: list, player: p.id }); }
function persistPlayer() { const p = currentPlayer(); savePlayer({ ...p, heightIn: st.heightIn, hand: { ...st.hand } }).catch(() => toast('Could not save the player. Try again.')); }
let appliedPlayer = '';
function applyPlayer(p) { st.playerId = p.id; st.heightIn = p.heightIn || 70; st.hand = { hit: (p.hand && p.hand.hit) || 'R', pitch: (p.hand && p.hand.pitch) || 'R' }; syncHand(); const hs = $('htSel'); if (hs) hs.value = String(st.heightIn); appliedPlayer = JSON.stringify(p); }
function syncPlayers() {
  const p = currentPlayer(); if (!p.unsaved && JSON.stringify(p) !== appliedPlayer) applyPlayer(p);
  const sel = $('plSel'); if (!sel || document.activeElement === sel) return;
  const list = players().length ? players() : [p];
  sel.innerHTML = list.map(x => `<option value="${esc(x.id)}">${esc(x.name)}</option>`).join('') + '<option value="__add">+ Add player…</option>';
  sel.value = p.id;
}
$('plSel').onchange = e => {
  const v = e.target.value;
  if (v === '__add') { e.target.value = currentPlayer().id; $('addPl').hidden = false; $('plName').value = ''; $('plName').focus(); return; }
  const p = players().find(x => x.id === v); if (!p) return;
  applyPlayer(p); store.setPrefs({ player: p.id }).catch(() => {});
  if (st.source === 'video' && !st.saved) { st.name = p.name + ' · ' + MODE[st.mode].noun + ' · ' + fmtDate(Date.now()); renderReport(); }
};
$('plCancel').onclick = () => { $('addPl').hidden = true; };
$('addPl').onsubmit = async e => {
  e.preventDefault(); const name = $('plName').value.trim(); if (!name) { toast('Type a name for the player.'); return; }
  const cur = currentPlayer(); if (cur.unsaved) { try { await savePlayer({ ...cur, heightIn: st.heightIn, hand: { ...st.hand } }); } catch {} }
  const p = { id: 'p' + Date.now().toString(36), name, heightIn: st.heightIn, hand: { ...st.hand } };
  try { await savePlayer(p); applyPlayer(p); syncPlayers(); $('addPl').hidden = true; toast(`Added ${name}. Set ${name}'s height and ${MODE[st.mode].hand.toLowerCase()} side before filming.`); } catch { toast('Could not add the player. Try again.'); }
};

/*COMPARE*/
/*CAMERA*/

// boot
syncPlayers();
setMode(st.mode);
store.init();
requestAnimationFrame(loop);
setTimeout(() => { ensureModel().catch(() => {}); }, 1500);
setTimeout(() => { ensurePrecise().catch(() => {}); ensureDetector(); }, 4000);
})();
