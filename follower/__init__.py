"""Pose follower with three two-handed gestures: stop, resume and a pose-
diagram show/hide toggle.

Sub-packages (kept deliberately separate):

- ``follower.logic``  - pure-Python detection & decision logic. Never imports
  cv2 or mediapipe, so it can be unit-tested anywhere.
- ``follower.vision`` - everything that touches cv2 / mediapipe: camera and
  video capture, running the MediaPipe models, drawing the pose diagram, and
  the main loop that wires it all together.

Test-only code lives outside the package, in ``tests/``.
"""
