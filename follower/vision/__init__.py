"""Everything that touches cv2 / mediapipe: capture, model inference, drawing,
and the main loop. Decisions are delegated to ``follower.logic``."""
import os

# Quieten MediaPipe/TFLite C++ info logging (must be set before mediapipe is imported).
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
