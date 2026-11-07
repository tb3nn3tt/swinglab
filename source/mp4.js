/* ---------------- tiny MP4/MOV demuxer: just enough to hand video samples to WebCodecs ----------------
   Some phone videos (HDR, HEVC) play in the browser but their pixels are locked away from scripts, so the
   tracker cannot read them. Decoding the file ourselves with WebCodecs gets around that. */
function mp4Demux(buf) {
  const dv = new DataView(buf), u8 = new Uint8Array(buf);
  const u32 = o => dv.getUint32(o), u16 = o => dv.getUint16(o), type = o => String.fromCharCode(u8[o], u8[o+1], u8[o+2], u8[o+3]);
  const kids = (s, e) => { const out = []; let o = s; while (o + 8 <= e) { let sz = u32(o); const t = type(o + 4); let h = 8; if (sz === 1) { sz = Number(dv.getBigUint64(o + 8)); h = 16; } else if (sz === 0) sz = e - o; if (sz < h || o + sz > e) break; out.push({ t, s: o + h, e: o + sz, o }); o += sz; } return out; };
  const find = (s, e, t) => kids(s, e).find(b => b.t === t);
  const path = (b, ...ts) => { for (const t of ts) { if (!b) return null; b = find(b.s, b.e, t); } return b; };
  const moov = kids(0, buf.byteLength).find(b => b.t === 'moov'); if (!moov) throw new Error('no moov');
  for (const trak of kids(moov.s, moov.e).filter(b => b.t === 'trak')) {
    const hdlr = path(trak, 'mdia', 'hdlr'); if (!hdlr || type(hdlr.s + 8) !== 'vide') continue;
    const tkhd = path(trak, 'tkhd'), tv = u8[tkhd.s]; const mo = tkhd.s + (tv === 1 ? 4 + 8 + 8 + 4 + 4 + 8 : 4 + 4 + 4 + 4 + 4 + 4) + 8 + 2 + 2 + 2 + 2;
    const fx = o => dv.getInt32(o)/65536; const a = fx(mo), b = fx(mo + 4); const rotation = Math.round(Math.atan2(b, a)*180/Math.PI);
    const mdhd = path(trak, 'mdia', 'mdhd'), mv = u8[mdhd.s]; const timescale = mv === 1 ? u32(mdhd.s + 20) : u32(mdhd.s + 12);
    const stbl = path(trak, 'mdia', 'minf', 'stbl');
    const stsd = find(stbl.s, stbl.e, 'stsd'), ent = kids(stsd.s + 8, stsd.e)[0];
    const fourcc = ent.t, width = u16(ent.s + 24), height = u16(ent.s + 26);
    const cfgs = kids(ent.s + 78, ent.e), cfg = cfgs.find(c => ['avcC', 'hvcC', 'vpcC', 'av1C'].includes(c.t));
    const description = cfg && (cfg.t === 'avcC' || cfg.t === 'hvcC' || cfg.t === 'av1C') ? u8.slice(cfg.s, cfg.e) : undefined;
    const hex = (v, n = 2) => v.toString(16).toUpperCase().padStart(n, '0');
    let codec = fourcc;
    if (cfg && cfg.t === 'avcC') codec = 'avc1.' + hex(u8[cfg.s + 1]) + hex(u8[cfg.s + 2]) + hex(u8[cfg.s + 3]);
    else if (cfg && cfg.t === 'hvcC') { const c = cfg.s, ps = u8[c + 1], space = ['', 'A', 'B', 'C'][ps >> 6], tier = (ps >> 5) & 1 ? 'H' : 'L', prof = ps & 31;
      let comp = dv.getUint32(c + 2), rev = 0; for (let i = 0; i < 32; i++) { rev = (rev << 1) | (comp & 1); comp >>>= 1; } rev >>>= 0;
      const lvl = u8[c + 12]; const cons = []; for (let i = 0; i < 6; i++) cons.push(u8[c + 6 + i]); while (cons.length && cons[cons.length - 1] === 0) cons.pop();
      codec = [fourcc === 'hev1' ? 'hev1' : 'hvc1', space + prof, rev.toString(16).toUpperCase(), tier + lvl].concat(cons.map(x => x.toString(16).toUpperCase())).join('.'); }
    else if (cfg && cfg.t === 'vpcC') { const c = cfg.s + 4; codec = `vp09.${String(u8[c]).padStart(2, '0')}.${String(u8[c + 1]).padStart(2, '0')}.${String(u8[c + 2] >> 4).padStart(2, '0')}`; }
    else if (cfg && cfg.t === 'av1C') { const c = cfg.s; const prof = u8[c + 1] >> 5, lvl = u8[c + 1] & 31, tier = (u8[c + 2] >> 7) ? 'H' : 'M', hb = (u8[c + 2] >> 6) & 1, tw = (u8[c + 2] >> 5) & 1; codec = `av01.${prof}.${String(lvl).padStart(2, '0')}${tier}.${hb ? (tw ? '12' : '10') : '08'}`; }
    // colour tags (nclx), so HDR clips get the same tone mapping the browser's player gives them
    let colorSpace; const colr = cfgs.find(c => c.t === 'colr');
    if (colr && type(colr.s) === 'nclx') { const P = { 1: 'bt709', 5: 'bt470bg', 6: 'smpte170m', 9: 'bt2020', 12: 'smpte432' }, T = { 1: 'bt709', 6: 'smpte170m', 8: 'linear', 13: 'iec61966-2-1', 16: 'pq', 18: 'hlg' }, M = { 0: 'rgb', 1: 'bt709', 5: 'bt470bg', 6: 'smpte170m', 9: 'bt2020-ncl' };
      const pr = u16(colr.s + 4), tf = u16(colr.s + 6), mx = u16(colr.s + 8), fr = !!(u8[colr.s + 10] >> 7);
      colorSpace = { primaries: P[pr] || null, transfer: T[tf] || null, matrix: M[mx] || null, fullRange: fr }; }
    const tbl = t => find(stbl.s, stbl.e, t);
    const stsz = tbl('stsz'), fixed = u32(stsz.s + 4), n = u32(stsz.s + 8), sizes = new Uint32Array(n); for (let i = 0; i < n; i++) sizes[i] = fixed || u32(stsz.s + 12 + i*4);
    const co = tbl('stco') || tbl('co64'), nc = u32(co.s + 4), chunk = []; for (let i = 0; i < nc; i++) chunk.push(co.t === 'co64' ? Number(dv.getBigUint64(co.s + 8 + i*8)) : u32(co.s + 8 + i*4));
    const stsc = tbl('stsc'), ns = u32(stsc.s + 4), sc = []; for (let i = 0; i < ns; i++) sc.push([u32(stsc.s + 8 + i*12), u32(stsc.s + 12 + i*12)]);
    const offs = new Float64Array(n); { let si = 0; for (let ci = 0; ci < nc && si < n; ci++) { let per = sc[0][1]; for (const [first, spc] of sc) if (ci + 1 >= first) per = spc; let o = chunk[ci]; for (let k = 0; k < per && si < n; k++) { offs[si] = o; o += sizes[si]; si++; } } }
    const stts = tbl('stts'), dts = new Float64Array(n); { let t = 0, si = 0; for (let i = 0, m = u32(stts.s + 4); i < m; i++) { const cnt = u32(stts.s + 8 + i*8), d = u32(stts.s + 12 + i*8); for (let k = 0; k < cnt && si < n; k++) { dts[si++] = t; t += d; } } }
    const cts = Float64Array.from(dts); const ctts = tbl('ctts');
    if (ctts) { const v = u8[ctts.s]; let si = 0; for (let i = 0, m = u32(ctts.s + 4); i < m; i++) { const cnt = u32(ctts.s + 8 + i*8), off = v === 1 ? dv.getInt32(ctts.s + 12 + i*8) : u32(ctts.s + 12 + i*8); for (let k = 0; k < cnt && si < n; k++) cts[si++] += off; } }
    const stss = tbl('stss'), key = new Uint8Array(n); if (stss) { for (let i = 0, m = u32(stss.s + 4); i < m; i++) key[u32(stss.s + 8 + i*4) - 1] = 1; } else key.fill(1);
    let c0 = Infinity; for (let i = 0; i < n; i++) c0 = Math.min(c0, cts[i]);
    const samples = []; for (let i = 0; i < n; i++) samples.push({ i, off: offs[i], size: sizes[i], key: !!key[i], t: (cts[i] - c0)/timescale, dur: i + 1 < n ? (dts[i+1] - dts[i])/timescale : 0 });
    const durTotal = n ? (dts[n-1] - dts[0])/timescale + (samples[n-2] ? samples[n-2].dur : 0) : 0;
    return { codec, fourcc, width, height, rotation, description, colorSpace, samples, fps: n > 1 ? (n - 1)/((dts[n-1] - dts[0])/timescale) : 30, duration: durTotal };
  }
  throw new Error('no video track');
}

// sequential frame reader on top of WebCodecs; frameAt(t) draws the frame showing at time t into a canvas the size the video displays at
function makeFrameReader(file, info) {
  const S = info.samples, byT = S.slice().sort((a, b) => a.t - b.t);
  const rot = ((info.rotation % 360) + 360) % 360, sideways = rot === 90 || rot === 270;
  const cv = document.createElement('canvas'); cv.width = sideways ? info.height : info.width; cv.height = sideways ? info.width : info.height;
  const cx = cv.getContext('2d', { willReadFrequently: true });
  let dec = null, next = 0, got = new Map(), err = null, gen = 0;
  const reset = kf => { if (dec) { try { dec.close(); } catch {} } for (const f of got.values()) f.close(); got = new Map(); err = null; const g = ++gen;
    dec = new VideoDecoder({ output: f => { if (g !== gen) { f.close(); return; } got.set(Math.round(f.timestamp/1000), f); if (got.size > 24) { const k = Math.min(...got.keys()); got.get(k).close(); got.delete(k); } }, error: e => { err = e; } });
    dec.configure({ codec: info.codec, codedWidth: info.width, codedHeight: info.height, description: info.description, ...(info.colorSpace ? { colorSpace: info.colorSpace } : {}) }); next = kf; };
  const read = async s => new Uint8Array(await file.slice(s.off, s.off + s.size).arrayBuffer());
  async function frameAt(t) {
    let k = 0; for (let i = 0; i < byT.length; i++) { if (byT[i].t <= t + 1e-4) k = i; else break; }
    const s = byT[k], ms = Math.round(s.t*1e3);
    if (got.has(ms)) return draw(got.get(ms), ms);
    let kf = s.i; while (kf > 0 && !S[kf].key) kf--;
    if (!dec || dec.state !== 'configured' || next > s.i + 32 || kf > next) reset(kf);
    let fed = 0;
    for (let spin = 0; spin < 4000; spin++) {
      if (err) throw err;
      if (got.has(ms)) return draw(got.get(ms), ms);
      if (next < S.length && (next <= s.i + 16) && dec.decodeQueueSize < 6) { const q = S[next]; dec.decode(new EncodedVideoChunk({ type: q.key ? 'key' : 'delta', timestamp: Math.round(q.t*1e6), duration: Math.round(q.dur*1e6), data: await read(q) })); next++; fed++; continue; }
      if (next >= S.length || next > s.i + 16) { await dec.flush(); if (got.has(ms)) return draw(got.get(ms), ms); reset(kf); if (spin > 2000) break; continue; }
      await new Promise(r => setTimeout(r, 2));
    }
    throw new Error('frame not decoded');
  }
  function draw(f, ms) {
    cx.save(); cx.setTransform(1, 0, 0, 1, 0, 0);
    if (rot === 90) { cx.translate(cv.width, 0); cx.rotate(Math.PI/2); }
    else if (rot === 180) { cx.translate(cv.width, cv.height); cx.rotate(Math.PI); }
    else if (rot === 270) { cx.translate(0, cv.height); cx.rotate(-Math.PI/2); }
    cx.drawImage(f, 0, 0, info.width, info.height); cx.restore();
    for (const [k, v] of got) if (k < ms) { v.close(); got.delete(k); }
    return cv;
  }
  return { canvas: cv, frameAt, close: () => { if (dec) try { dec.close(); } catch {} for (const f of got.values()) f.close(); got.clear(); } };
}
if (typeof module !== 'undefined') module.exports = { mp4Demux };
