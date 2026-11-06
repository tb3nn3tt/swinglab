# SwingLab

Phone web app that films a baseball swing or pitch, tracks every joint and the bat, scores the mechanics, shows a corrected skeleton, and compares reps and players side by side.

Open it at **https://tb3nn3tt.github.io/swinglab/** in Safari or Chrome, then add it to your home screen.

- `index.html`, `models/`, `icons/`, `manifest.webmanifest`, `sw.js`: the live site
- `source/`: the source files the site is built from

Pose tracking uses MoveNet SinglePose Lightning (TensorFlow.js). Everything runs on the phone; videos are never uploaded.
