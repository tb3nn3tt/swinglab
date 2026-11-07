/* ---------------- live camera: preview with skeleton, record in the app (standalone build) ---------------- */
const cam = { stream: null, rec: null, chunks: [], facing: 'environment', live: false, t0: 0, crop: null, posing: false, wake: null, timerI: 0, autoStop: 0, delayT: 0, discard: false };
const camEl = $('cam'), camVid = $('camVid'), camCv = $('camCv');
function camStatus(t, ok) { const s = $('camStatus'); s.textContent = t; s.dataset.ok = ok ? '1' : ''; }
async function openCam() {
  if (st.busy) { toast('Wait for the current analysis to finish.'); return; }
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) { toast('This browser cannot open the camera. Use Upload a video instead.'); return; }
  camEl.hidden = false; document.documentElement.classList.add('cam-open');
  $('camHint').textContent = st.mode === 'hit' ? 'Face the hitter\'s chest from 15–20 ft. Whole body and bat in frame.' : 'Film from the side, 20–30 ft away. Whole body in frame.';
  camStatus('Opening camera…');
  const ok = await startStream(); if (!ok) return;
  camStatus('Loading tracker…');
  ensureModel().catch(() => camStatus('Tracker did not load. You can still record.'));
  cam.live = true; cam.crop = null; requestAnimationFrame(liveLoop);
  try { cam.wake = navigator.wakeLock ? await navigator.wakeLock.request('screen') : null; } catch {}
}
async function startStream() {
  stopStream();
  try {
    cam.stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: { facingMode: { ideal: cam.facing }, width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 120 } } });
  } catch (e) {
    camStatus(e && e.name === 'NotAllowedError' ? 'Camera access is blocked. Allow the camera for this site in your browser settings, then tap Record again.' : 'Could not open the camera. Close other apps using it and try again.');
    return false;
  }
  camVid.srcObject = cam.stream; camVid.muted = true; camVid.playsInline = true;
  try { await camVid.play(); } catch {}
  return true;
}
function stopStream() { if (cam.stream) cam.stream.getTracks().forEach(t => t.stop()); cam.stream = null; camVid.srcObject = null; }
function closeCam() {
  if (cam.rec && cam.rec.state === 'recording') { cam.discard = true; try { cam.rec.stop(); } catch {} }
  clearTimeout(cam.delayT); clearTimeout(cam.autoStop); clearInterval(cam.timerI); setRecUi(false);
  cam.live = false; stopStream(); camEl.hidden = true; document.documentElement.classList.remove('cam-open');
  if (cam.wake) { cam.wake.release().catch(() => {}); cam.wake = null; }
}
async function liveLoop() {
  if (!cam.live) return;
  if (model && camVid.videoWidth && !cam.posing) {
    cam.posing = true;
    try { const W = camVid.videoWidth, H = camVid.videoHeight; const r = await poseAt(W, H, cam.crop, camVid); cam.crop = r.crop; drawCamPose(r.k, W, H); } catch {}
    cam.posing = false;
  }
  requestAnimationFrame(liveLoop);
}
function drawCamPose(k, W, H) {
  const { ctx, w, h } = canvasCtx(camCv); ctx.clearRect(0, 0, w, h);
  const fit = videoFit(W, H, w, h), f = new Float32Array(34);
  for (let j = 0; j < 17; j++) { f[j*2] = k[j*3]; f[j*2+1] = k[j*3+1]; }
  const seen = j => k[j*3+2] > 0.3;
  const body = [5, 6, 11, 12, 13, 14].filter(seen).length;
  if (body >= 4) drawSkel(ctx, f, (x, y) => [fit.x + x*fit.s, fit.y + y*fit.s], { colors: { l: css('--lead'), b: css('--back'), c: '#E9EEF4' }, scale: fit.s*H*0.6, dim: k, leadLeft: st.hand[st.mode] === 'R' });
  if (cam.rec && cam.rec.state === 'recording') return;
  if (body < 4) camStatus('No player found yet. Point the camera at the whole body.');
  else if (!(seen(0) || seen(1) || seen(2)) || !(seen(15) || seen(16))) camStatus('Step back or zoom out so head to feet are in frame.');
  else camStatus('Ready. Whole body in frame.', true);
}
function pickMime() {
  if (!window.MediaRecorder || !MediaRecorder.isTypeSupported) return '';
  for (const m of ['video/mp4;codecs=avc1', 'video/mp4', 'video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm']) if (MediaRecorder.isTypeSupported(m)) return m;
  return '';
}
function setRecUi(on) {
  const b = $('camRec'); b.classList.toggle('on', on); b.setAttribute('aria-label', on ? 'Stop recording' : 'Start recording');
  $('camFlip').disabled = on; clearInterval(cam.timerI);
  if (on) { const t = $('camTimer'); cam.timerI = setInterval(() => { const s = Math.floor((performance.now() - cam.t0)/1000); t.textContent = '0:' + String(s).padStart(2, '0'); }, 250); camStatus('Recording. Take the ' + MODE[st.mode].noun + ', then tap stop.', true); }
  else $('camTimer').textContent = '0:00';
}
function startRec() {
  if (!cam.stream) return;
  if (!window.MediaRecorder) { toast('This browser cannot record video. Use Upload a video instead.'); return; }
  const mime = pickMime(); cam.chunks = []; cam.discard = false;
  try { cam.rec = new MediaRecorder(cam.stream, mime ? { mimeType: mime, videoBitsPerSecond: 8000000 } : undefined); }
  catch { cam.rec = new MediaRecorder(cam.stream); }
  cam.rec.ondataavailable = e => { if (e.data && e.data.size) cam.chunks.push(e.data); };
  cam.rec.onstop = () => {
    const dur = (performance.now() - cam.t0)/1000, type = (cam.rec && cam.rec.mimeType) || mime || 'video/webm';
    const blob = new Blob(cam.chunks, { type }); setRecUi(false);
    if (cam.discard || !blob.size) return;
    if (dur < 1) { toast('That clip was too short. Record from the setup through the finish.'); return; }
    closeCam(); setTab('Analyze');
    const file = new File([blob], MODE[st.mode].noun + (type.includes('mp4') ? '.mp4' : '.webm'), { type });
    analyzeFile(file, { slow: 1, duration: dur });
  };
  cam.rec.start(250); cam.t0 = performance.now(); setRecUi(true);
  cam.autoStop = setTimeout(stopRec, 20000);
}
function stopRec() { clearTimeout(cam.autoStop); if (cam.rec && cam.rec.state === 'recording') cam.rec.stop(); }
$('camBtn').onclick = openCam;
$('camClose').onclick = closeCam;
$('camFlip').onclick = async () => { cam.facing = cam.facing === 'environment' ? 'user' : 'environment'; cam.crop = null; await startStream(); };
$('camRec').onclick = () => {
  if (cam.rec && cam.rec.state === 'recording') { stopRec(); return; }
  if (cam.delayT) { clearTimeout(cam.delayT); cam.delayT = 0; camStatus('Countdown canceled.'); return; }
  if (!$('camDelay').checked) { startRec(); return; }
  let n = 3; camStatus('Recording in 3…', true);
  const tick = () => { n--; if (n <= 0) { cam.delayT = 0; startRec(); return; } camStatus('Recording in ' + n + '…', true); cam.delayT = setTimeout(tick, 1000); };
  cam.delayT = setTimeout(tick, 1000);
};
document.addEventListener('visibilitychange', () => { if (document.hidden && cam.live && !(cam.rec && cam.rec.state === 'recording')) closeCam(); });

/* ---------------- standalone extras: backup, install, offline ---------------- */
$('camBtn').hidden = false;
$('recBtn').classList.remove('primary'); $('recLbl').textContent = 'Upload video';
$('recLbl').dataset.standalone = '1';
$('recHint').innerHTML = '<b>Record</b> films right here with a live skeleton. <b>Upload video</b> takes a clip you already shot, including slo-mo (set Clip to match).';
$('storeTag').textContent = 'Saved on this phone';
$('libNote').textContent = 'Each rep stores the skeleton, phases and numbers, not the video. Reps are saved on this phone, so back up your library now and then.';
$('backupRow').hidden = false;
$('bkSave').onclick = () => {
  const data = JSON.stringify({ app: 'SwingLab', v: 1, exported: new Date().toISOString(), records: st.records, prefs: st.prefs });
  const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([data], { type: 'application/json' }));
  a.download = 'swinglab-backup-' + new Date().toISOString().slice(0, 10) + '.json'; document.body.appendChild(a); a.click();
  setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1000); toast('Backup saved to your downloads.');
};
$('bkIn').onchange = async e => {
  const f = e.target.files && e.target.files[0]; e.target.value = ''; if (!f) return;
  try {
    const d = JSON.parse(await f.text()); if (!d || !Array.isArray(d.records)) throw 0;
    const l = store.readLocal(); const have = new Set((l.records || []).map(r => r.id));
    const add = d.records.filter(r => r && r.id && r.kind && !have.has(r.id));
    l.records = (l.records || []).concat(add);
    const pl = (l.prefs && l.prefs.players) || []; const ids = new Set(pl.map(p => p.id));
    l.prefs = { ...(d.prefs || {}), ...(l.prefs || {}), players: pl.concat(((d.prefs && d.prefs.players) || []).filter(p => !ids.has(p.id))) };
    if (!store.writeLocal(l)) throw new Error('space');
    st.records = l.records; st.prefs = l.prefs; renderAll(); toast(`Restored ${add.length} rep${add.length === 1 ? '' : 's'}.`);
  } catch (err) { toast(err && err.message === 'space' ? 'This phone ran out of space for saved reps.' : 'That file is not a SwingLab backup.'); }
};
{ // install prompt
  const standaloneNow = matchMedia('(display-mode: standalone)').matches || navigator.standalone;
  let dismissed = false; try { dismissed = localStorage.getItem('swinglab.installHide') === '1'; } catch {}
  const card = $('installCard');
  if (!standaloneNow && !dismissed) {
    const ios = /iP(hone|ad|od)/.test(navigator.userAgent) || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
    $('installTxt').innerHTML = ios ? 'Install SwingLab on your home screen: tap <b>Share</b> in Safari, then <b>Add to Home Screen</b>. It opens full screen like a regular app.' : 'Install SwingLab on your home screen so it opens full screen like a regular app.';
    card.hidden = ios ? false : true;
    window.addEventListener('beforeinstallprompt', e => { e.preventDefault(); card.hidden = false; const b = $('installBtn'); b.hidden = false; b.onclick = async () => { e.prompt(); try { await e.userChoice; } catch {} card.hidden = true; }; });
  }
  $('installHide').onclick = () => { card.hidden = true; try { localStorage.setItem('swinglab.installHide', '1'); } catch {} };
}
if ('serviceWorker' in navigator && location.protocol === 'https:') {
  const hadSW = !!navigator.serviceWorker.controller;
  navigator.serviceWorker.register('sw.js').then(reg => { setInterval(() => reg.update().catch(() => {}), 30*60*1000); document.addEventListener('visibilitychange', () => { if (!document.hidden) reg.update().catch(() => {}); }); }).catch(() => {});
  // a new version took over: reload when nothing is in progress, so you never run stale code
  navigator.serviceWorker.addEventListener('controllerchange', () => {
    if (!hadSW) return;
    const go = () => { if (!st.busy && !cam.live) location.reload(); else setTimeout(go, 2000); };
    toast('SwingLab updated. Reloading…'); setTimeout(go, 800);
  });
}
