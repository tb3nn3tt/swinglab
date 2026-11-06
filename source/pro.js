/* ---------- pro model: a 3D swing / delivery built from published biomechanics norms ----------
   World: X toward the pitcher (hitting) or plate (pitching), Y up, Z toward the camera, which faces the
   player's chest (hitting) or films from the side (pitching). Segment lengths are fractions of height
   (Winter's anthropometric tables). Timing and rotation targets follow pro studies: pelvis peaks before
   torso before arms before bat/hand; ~150 ms from launch to contact; ~140 ms from foot plant to release. */
const v3 = { add: (a, b) => [a[0]+b[0], a[1]+b[1], a[2]+b[2]], sub: (a, b) => [a[0]-b[0], a[1]-b[1], a[2]-b[2]], mul: (a, s) => [a[0]*s, a[1]*s, a[2]*s],
  dot: (a, b) => a[0]*b[0] + a[1]*b[1] + a[2]*b[2], len: a => Math.hypot(a[0], a[1], a[2]), cross: (a, b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]] };
v3.norm = a => { const l = v3.len(a) || 1; return [a[0]/l, a[1]/l, a[2]/l]; };
const rotY = (v, deg) => { const r = deg/DEG, c = Math.cos(r), s = Math.sin(r); return [v[0]*c + v[2]*s, v[1], -v[0]*s + v[2]*c]; };
function pchip(ts, vs) {
  const n = ts.length, h = [], d = [], m = new Array(n).fill(0);
  for (let i = 0; i < n - 1; i++) { h[i] = ts[i+1] - ts[i]; d[i] = (vs[i+1] - vs[i])/h[i]; }
  for (let i = 1; i < n - 1; i++) { if (d[i-1]*d[i] <= 0) continue; const w1 = 2*h[i] + h[i-1], w2 = h[i] + 2*h[i-1]; m[i] = (w1 + w2)/(w1/d[i-1] + w2/d[i]); }
  return t => {
    if (t <= ts[0]) return vs[0]; if (t >= ts[n-1]) return vs[n-1];
    let i = 0; while (t > ts[i+1]) i++;
    const hh = h[i], s = (t - ts[i])/hh, s2 = s*s, s3 = s2*s;
    return (2*s3 - 3*s2 + 1)*vs[i] + (s3 - 2*s2 + s)*hh*m[i] + (-2*s3 + 3*s2)*vs[i+1] + (s3 - s2)*hh*m[i+1];
  };
}
const curve = keys => { const ts = keys.map(k => k[0]); if (Array.isArray(keys[0][1])) { const fs = keys[0][1].map((_, c) => pchip(ts, keys.map(k => k[1][c]))); return t => fs.map(f => f(t)); } return pchip(ts, keys.map(k => k[1])); };
function ik(a, c, l1, l2, pole) {
  let d = v3.sub(c, a), L = v3.len(d); const maxL = (l1 + l2)*0.995;
  if (L > maxL) { c = v3.add(a, v3.mul(d, maxL/L)); d = v3.sub(c, a); L = maxL; }
  const dn = v3.mul(d, 1/L), x = (l1*l1 - l2*l2 + L*L)/(2*L), hh = Math.sqrt(Math.max(0, l1*l1 - x*x));
  let p = v3.sub(pole, v3.mul(dn, v3.dot(pole, dn))); p = v3.norm(p);
  return { mid: v3.add(v3.add(a, v3.mul(dn, x)), v3.mul(p, hh)), end: c };
}
const SEG = { hip: 0.085, sh: 0.112, spine: 0.29, thigh: 0.245, shank: 0.246, upper: 0.186, fore: 0.146 };
// pose at one instant -> 17 joints projected to the camera (x = X, y = -Y) plus 3D extras
function poseFrom(p) {
  const P = p.pelvis, up = v3.norm([-Math.tan(p.tiltBack/DEG), 1, Math.tan(p.tiltZ/DEG)]);
  const lHip = v3.add(P, rotY([SEG.hip, 0, 0], p.pelvisRot)), bHip = v3.add(P, rotY([-SEG.hip, 0, 0], p.pelvisRot));
  const Sc = v3.add(P, v3.mul(up, SEG.spine));
  let lat = rotY([1, 0, 0], p.torsoRot); lat = v3.norm(v3.sub(lat, v3.mul(up, v3.dot(lat, up))));
  const st = (p.shTilt || 0)/DEG; lat = v3.norm(v3.add(v3.mul(lat, Math.cos(st)), v3.mul(up, Math.sin(st))));
  const lSh = v3.add(Sc, v3.mul(lat, SEG.sh)), bSh = v3.sub(Sc, v3.mul(lat, SEG.sh));
  const Hc = v3.add(Sc, v3.mul(v3.norm(v3.add(v3.mul(up, 0.4), [0, 0.6, 0])), 0.125));
  const pitchR = (p.headDown || 0)/DEG; let f = rotY([0, 0, 1], p.headYaw); f = v3.norm([f[0]*Math.cos(pitchR), -Math.sin(pitchR), f[2]*Math.cos(pitchR)]);
  const right = v3.norm(v3.cross(f, [0, 1, 0]));
  const nose = v3.add(v3.add(Hc, v3.mul(f, 0.065)), [0, -0.012, 0]);
  const eyeC = v3.add(v3.add(Hc, v3.mul(f, 0.05)), [0, 0.012, 0]);
  const lEye = v3.sub(eyeC, v3.mul(right, 0.032)), rEye = v3.add(eyeC, v3.mul(right, 0.032));
  const lEar = v3.sub(v3.sub(Hc, v3.mul(right, 0.072)), v3.mul(f, 0.01)), rEar = v3.sub(v3.add(Hc, v3.mul(right, 0.072)), v3.mul(f, 0.01));
  const lLeg = ik(lHip, p.lAnkle, SEG.thigh, SEG.shank, p.lKneePole), bLeg = ik(bHip, p.bAnkle, SEG.thigh, SEG.shank, p.bKneePole);
  const lArm = ik(lSh, p.lWrist, SEG.upper, SEG.fore, p.lElbowPole), bArm = ik(bSh, p.bWrist, SEG.upper, SEG.fore, p.bElbowPole);
  // RHB / RHP: left side leads
  const J = [nose, lEye, rEye, lEar, rEar, lSh, bSh, lArm.mid, bArm.mid, lArm.end, bArm.end, lHip, bHip, lLeg.mid, bLeg.mid, lLeg.end, bLeg.end];
  const f2 = new Float32Array(34); J.forEach((q, j) => { f2[j*2] = q[0]; f2[j*2+1] = -q[1]; });
  return { f: f2, J };
}
function proKeys(kind, faults) {
  const F = faults || {};
  if (kind === 'hit') {
    // stance 0–0.5, load to 0.85, stride lands 0.95, launch 1.00, contact 1.15, extension 1.23, finish to 1.75
    const TFP = 0.95, TC = 1.15;
    const front = F.overstride ? 0.52 : 0.33;
    return { T: 1.85, fps: 60,
      pelvis: curve([[0, [0, 0.50, 0]], [0.5, [0, 0.50, 0]], [0.82, [-0.03, 0.495, 0]], [TFP, [0.03, 0.47 - (F.headDrop ? 0.02 : 0), 0]], [TC, [0.075, 0.468 - (F.headDrop ? 0.06 : 0), 0.01]], [1.4, [0.09, 0.47, 0]], [1.85, [0.08, 0.49, 0]]]),
      pelvisRot: curve([[0, 0], [0.5, 0], [0.84, -12], [TFP, 6], [1.04, 42], [1.10, 66], [TC, 80], [1.3, 98], [1.85, 104]]),
      torsoRot: curve(F.earlyOpen ? [[0, -4], [0.5, -6], [0.8, -14], [TFP, 32], [1.02, 55], [1.08, 78], [TC, 105], [1.3, 148], [1.85, 160]]
        : [[0, -4], [0.5, -6], [0.84, -26], [TFP, -22], [1.01, -6], [1.07, 30], [1.11, 62], [TC, 86], [1.3, 140], [1.85, 158]]),
      tiltBack: curve([[0, 4], [0.84, 6], [TFP, 9], [TC, 20 - (F.headDrop ? 6 : 0)], [1.4, 16], [1.85, 10]]),
      tiltZ: curve([[0, 22], [TFP, 24], [TC, 26 + (F.headDrop ? 14 : 0)], [1.5, 18], [1.85, 14]]),
      shTilt: curve([[0, 2], [0.84, 4], [TFP, 4], [TC, -14], [1.4, 10], [1.85, 14]]),
      headYaw: curve([[0, 82], [TC, 86], [1.4, 84], [1.85, 70]]), headDown: curve([[0, 8], [TFP, 10], [TC, 18], [1.35, 16], [1.85, 6]]),
      lAnkle: curve([[0, [0.15, 0.045, -0.02]], [0.5, [0.15, 0.045, -0.02]], [0.78, [0.11, 0.075, -0.03]], [0.88, [front - 0.05, 0.07, -0.02]], [TFP, [front, 0.045, -0.02]], [1.85, [front, 0.045, -0.02]]]),
      bAnkle: curve([[0, [-0.15, 0.045, 0]], [TFP, [-0.15, 0.045, 0]], [TC, [-0.135 + (F.backSlide ? 0.09 : 0), 0.07, 0.01]], [1.4, [-0.11 + (F.backSlide ? 0.09 : 0), 0.09, 0.02]], [1.85, [-0.1 + (F.backSlide ? 0.09 : 0), 0.09, 0.02]]]),
      lKneePole: curve([[0, [0.1, 0, 1]], [0.84, [-0.5, 0, 1]], [TFP, [0.3, 0, 1]], [1.85, [0.6, 0, 0.6]]]), bKneePole: curve([[0, [-0.1, 0, 1]], [TC, [0.6, 0, 0.8]], [1.85, [0.8, 0, 0.4]]]),
      grip: curve([[0, [-0.14, 0.80, 0.10]], [0.5, [-0.14, 0.80, 0.10]], [0.84, [-0.21, 0.80, 0.07]], [TFP, [-0.21, 0.79, 0.07]], [1.01, [-0.19, 0.76, 0.09]],
        [1.08, F.cast ? [-0.02, 0.70, 0.30] : [-0.07, 0.69, 0.16]], [TC, [0.08, 0.63, 0.20]], [1.23, [0.22, 0.65, 0.12]], [1.4, [0.10, 0.84, -0.12]], [1.85, [0.0, 0.88, -0.16]]]),
      batAz: curve([[0, 197], [0.5, 197], [0.84, 205], [TFP, 210], [1.02, 206], [1.08, 160], [TC, 92], [1.23, 15], [1.4, -95], [1.85, -140]]),
      batEl: curve(F.dump ? [[0, 58], [0.84, 50], [TFP, 44], [1.02, 5], [1.08, -38], [TC, -18], [1.23, -6], [1.4, 18], [1.85, 34]]
        : [[0, 58], [0.84, 50], [TFP, 45], [1.02, 30], [1.08, 6], [TC, -14], [1.23, -5], [1.4, 18], [1.85, 34]]),
      lElbowPole: curve([[0, [0.2, -1, 0.3]], [TC, [0.2, -1, 0.4]], [1.4, [0.3, -0.6, 0.6]], [1.85, [0, -1, 0.3]]]),
      bElbowPole: curve([[0, [-0.5, -1, 0]], [TFP, [-0.4, -1, -0.1]], [TC, [0, -1, 0.1]], [1.4, [0.4, -0.5, 0.5]], [1.85, [0.3, -0.8, 0.3]]]),
    };
  }
  // pitching: set 0–0.6, leg lift peak 1.10, foot plant 1.55, arm cocked 1.64, release 1.69, follow-through to 2.15
  const TL = 1.10, TFP = 1.55, TR = 1.69;
  const stride = F.shortStride ? 0.66 : 0.84;
  return { T: 2.2, fps: 60, pitch: true,
    pelvis: curve([[0, [0, 0.53, 0]], [0.6, [0, 0.53, 0]], [TL, [-0.01, 0.53, 0]], [1.35, [0.20, 0.46, 0]], [TFP, [0.44, 0.39, 0]], [TR, [0.56, 0.43, -0.02]], [1.9, [0.62, 0.42, -0.05]], [2.2, [0.64, 0.46, -0.06]]]),
    pelvisRot: curve([[0, 0], [0.6, 0], [TL, -18], [1.4, -8], [TFP, F.earlyHips ? 55 : 32], [1.61, 70], [TR, 90], [1.9, 112], [2.2, 118]]),
    torsoRot: curve(F.earlyHips ? [[0, 0], [0.6, 0], [TL, -20], [1.4, -10], [TFP, 30], [1.6, 70], [TR, 100], [1.9, 140], [2.2, 150]]
      : [[0, 0], [0.6, 0], [TL, -22], [1.4, -22], [TFP, -16], [1.6, 30], [1.65, 70], [TR, 100], [1.9, 140], [2.2, 150]]),
    tiltBack: curve([[0, 0], [TL, 2], [TFP, 4], [1.62, -10], [TR, F.upright ? -10 : -38], [1.9, -50], [2.2, -42]]),
    tiltZ: curve([[0, 0], [TFP, 4], [TR, 8], [2.2, 10]]),
    shTilt: curve([[0, 0], [TL, 0], [TFP, 6], [TR, -18], [2.2, -8]]),
    headYaw: curve([[0, 88], [2.2, 90]]), headDown: curve([[0, 2], [TR, 8], [1.9, 20], [2.2, 14]]),
    lAnkle: curve([[0, [0.12, 0.045, 0.0]], [0.6, [0.12, 0.045, 0.0]], [0.8, [0.08, 0.12, 0.0]], [TL, [0.16, 0.30, 0.02]], [1.35, [0.42, 0.18, 0.0]], [1.5, [stride - 0.04, 0.07, -0.01]], [TFP, [stride, 0.045, -0.01]], [2.2, [stride, 0.045, -0.01]]]),
    bAnkle: curve([[0, [0, 0.045, 0]], [1.4, [0, 0.045, 0]], [TFP, [0.03, 0.06, 0]], [TR, [0.14, 0.12, 0.02]], [1.9, [0.48, 0.22, 0.05]], [2.2, [0.70, 0.07, 0.06]]]),
    lKneePole: curve([[0, [0.5, 0, 0.8]], [TL, [1, 0.6, 0.2]], [TFP, [1, 0.1, 0.2]], [2.2, [1, 0, 0.2]]]), bKneePole: curve([[0, [0.6, 0, 0.8]], [TFP, [1, 0, 0.3]], [1.9, [1, 0.2, 0.2]], [2.2, [1, 0, 0.2]]]),
    // throwing hand path (absolute world positions) and glove hand
    bWrist: curve([[0, [0.04, 0.70, 0.13]], [0.6, [0.04, 0.70, 0.13]], [TL, [0.02, 0.73, 0.12]], [1.22, [0.06, 0.50, 0.12]], [1.38, [-0.02, 0.58, 0.04]],
      [TFP, F.lateArm ? [0.26, 0.60, 0.0] : [0.24, 0.88, -0.02]], [1.64, [0.45, 0.86, -0.08]], [TR, [0.92, 0.88, -0.08]], [1.78, [0.98, 0.56, -0.1]], [1.95, [0.86, 0.32, -0.16]], [2.2, [0.80, 0.40, -0.16]]]),
    lWrist: curve([[0, [0.04, 0.70, 0.13]], [0.6, [0.04, 0.70, 0.13]], [TL, [0.02, 0.73, 0.12]], [1.25, [0.26, 0.74, 0.12]], [TFP, [0.80, 0.76, 0.08]],
      [TR, F.gloveOut ? [0.86, 0.64, -0.2] : [0.70, 0.66, 0.08]], [1.9, [0.72, 0.56, 0.02]], [2.2, [0.74, 0.62, 0.02]]]),
    lElbowPole: curve([[0, [0, -1, 0.5]], [TFP, [0, -1, 0.4]], [TR, [0, -1, 0.2]], [2.2, [0, -1, 0.3]]]),
    bElbowPole: curve([[0, [0, -1, 0.5]], [1.3, [-0.6, -0.8, 0]], [TFP, [-1, -0.2, -0.2]], [1.64, [-0.6, -0.1, -0.6]], [TR, [0.2, -0.4, -0.8]], [2.2, [0, -1, -0.3]]]),
  };
}
function buildPro(kind, faults) {
  const K = proKeys(kind, faults), n = Math.round(K.T*K.fps), frames = [], tips = [];
  for (let i = 0; i < n; i++) {
    const t = i/K.fps, p = {};
    for (const k of Object.keys(K)) if (typeof K[k] === 'function') p[k] = K[k](t);
    if (kind === 'hit') {
      const az = p.batAz/DEG, el = p.batEl/DEG, B = [Math.cos(el)*Math.cos(az), Math.sin(el), Math.cos(el)*Math.sin(az)];
      p.lWrist = v3.sub(p.grip, v3.mul(B, 0.03)); p.bWrist = v3.add(p.grip, v3.mul(B, 0.045));
      const r = poseFrom(p); frames.push(r.f);
      const tip = v3.add(p.grip, v3.mul(B, 0.40)); tips.push({ x: tip[0], y: -tip[1], c: 1 });
    } else frames.push(poseFrom(p).f);
  }
  const seq = { frames, conf: frames.map(() => new Float32Array(NJ).fill(1)), fps: K.fps, leadLeft: true, H: 1, groundY: 0, originX: 0, fwd: 1, demo: true, pro: !faults };
  if (kind === 'hit') { seq.batTips = tips; addBatSpeed(seq); }
  return seq;
}
function buildDemo(kind) { return buildPro(kind); }
function buildFaulty(kind) { return buildPro(kind, kind === 'hit' ? { overstride: 1, headDrop: 1, earlyOpen: 1, dump: 1, cast: 1, backSlide: 1 } : { shortStride: 1, lateArm: 1, earlyHips: 1, upright: 1, gloveOut: 1 }); }
