"""Pose follower with gesture toggle - entry point.

- Locks onto the first person MediaPipe detects and draws their pose "sticks".
- Holding an OPEN PALM in one hand and a FIST in the other for 3 seconds stops
  following. Holding an OPEN PALM and SCISSORS (index + middle finger out,
  thumb tucked) for 3 seconds resumes it - locking onto whoever made the
  gesture. Holding a FIST and SCISSORS for 3 seconds shows/hides the pose
  diagram on the body, without affecting tracking. Each gesture must be
  lowered before it counts again.
- A header at the top of the image always shows which of these are currently
  available and what each one will do.

Uses the live camera when one is available; otherwise runs on the recordings
in tests/scenarios/. ``--record`` saves a live session (video + MediaPipe
landmarks + actions log) into outputs/. See ``python follow_pose.py --help``.

Keys: q / ESC = quit.

Code layout:
  follower/logic/   gesture + follow/stop logic (pure Python, no cv2/mediapipe)
  follower/vision/  camera/video capture, MediaPipe inference, drawing, main loop
  tests/            test-only code and recorded scenarios
"""
import sys

from follower.vision.app import main

if __name__ == "__main__":
    sys.exit(main())
