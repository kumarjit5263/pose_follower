"""Runs the MediaPipe Pose and Hands models on BGR frames.

Beyond plain MediaPipe, two things make detection match what the follow logic needs:

* **Pose-guided hand recovery.** MediaPipe Hands finds hands with a full-frame
  palm detector, which often misses a closed hand (or any smallish hand) - in a
  recorded live session one hand of the gesture went undetected for 6-8 s
  until the person stepped closer. When the pose shows a wrist inside the frame
  with no detected hand near it, we crop a square around that wrist, enlarge it
  and run the hand model on the crop, where the hand is big enough to be found
  (the same idea as MediaPipe Holistic's pose-derived hand regions).

* **Locking onto a given person.** The legacy Pose model follows one person and,
  after a reset, picks whoever is most prominent. ``lock_on(target)`` makes it
  pick the person who made the resume gesture: until a pose is found inside the
  target region, pose runs on a copy of the frame with everything outside that
  region greyed out. Once found, normal frames resume and MediaPipe's
  frame-to-frame tracking keeps following that person.
"""
import math
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2

from follower.logic import FrameRecord, Region, ResumeTarget, pose_matches_target

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
MEDIAPIPE_VERSION = mp.__version__

# pose landmark indices: (wrist, index finger, pinky) per side
_POSE_HANDS = ((15, 19, 17), (16, 20, 18))
_LEFT_HIP, _RIGHT_HIP = 23, 24
_SHOULDERS = (11, 12)

ROI_SIZE = 256           # px the wrist crop is scaled to for the hand model
LOCK_TIMEOUT_FRAMES = 30  # give up locking onto the gesturer after this many frames
MASK_VALUE = 114         # grey used to blank out everything outside the lock region


@dataclass
class Detection:
    """MediaPipe results for one frame (as protos, which the drawing utils need)."""
    pose_landmarks: Optional[object] = None            # NormalizedLandmarkList, or None
    hand_landmarks: List[object] = field(default_factory=list)  # NormalizedLandmarkList per hand
    recovered_hands: int = 0                           # of those, found via pose-guided crops
    lock_region: Optional[Region] = None               # set while locking onto a gesturer

    @property
    def pose_lm(self):
        """The 33 pose landmarks (for follower.logic), or None."""
        return self.pose_landmarks.landmark if self.pose_landmarks else None

    @property
    def hands(self):
        """List of 21-landmark hands (for follower.logic)."""
        return [h.landmark for h in self.hand_landmarks]

    @classmethod
    def from_record(cls, rec: FrameRecord) -> "Detection":
        """Rebuild a Detection from a frame of a ``.landmarks.jsonl`` recording."""
        pose = None
        if rec.pose is not None:
            pose = landmark_pb2.NormalizedLandmarkList(landmark=[
                landmark_pb2.NormalizedLandmark(x=p.x, y=p.y, z=p.z, visibility=p.visibility)
                for p in rec.pose])
        hands = [landmark_pb2.NormalizedLandmarkList(landmark=[
            landmark_pb2.NormalizedLandmark(x=p.x, y=p.y, z=p.z) for p in hand])
            for hand in rec.hands]
        return cls(pose, hands)


def hand_roi(pose_lm, side_index: int, width: int, height: int):
    """Square crop (cx, cy, size) in pixels likely to contain the hand at a pose
    wrist, or None if that wrist isn't usable (not visible, at the frame edge, or
    below the hips - i.e. the hand is down)."""
    wi, ii, pi = _POSE_HANDS[side_index]
    w = pose_lm[wi]
    if w.visibility < 0.5 or not (0.02 <= w.x <= 0.98 and 0.02 <= w.y <= 0.95):
        return None
    hips = [pose_lm[i] for i in (_LEFT_HIP, _RIGHT_HIP)
            if pose_lm[i].visibility >= 0.5 and pose_lm[i].y <= 1.0]
    if hips and w.y > max(h.y for h in hips):
        return None
    wx, wy = w.x * width, w.y * height
    # direction wrist -> fingers, from the pose's index / pinky points
    fx = (pose_lm[ii].x + pose_lm[pi].x) / 2 * width - wx
    fy = (pose_lm[ii].y + pose_lm[pi].y) / 2 * height - wy
    reach = math.hypot(fx, fy)
    l, r = (pose_lm[i] for i in _SHOULDERS)
    shoulders = math.hypot((l.x - r.x) * width, (l.y - r.y) * height) \
        if min(l.visibility, r.visibility) >= 0.5 else 0.0
    size = max(1.2 * shoulders, 4.0 * reach, 0.12 * height)
    return wx + 0.6 * fx, wy + 0.6 * fy, size


def crop_square(img, cx, cy, size, out_size=ROI_SIZE):
    """Crop a size x size square centred on (cx, cy), padding outside the image,
    and resize it. Returns (crop, x0, y0, side): (x0, y0) is the crop's top-left
    in ``img`` and ``side`` its size in pixels before resizing."""
    h, w = img.shape[:2]
    s = max(int(round(size)), 2)
    x0, y0 = int(round(cx - s / 2)), int(round(cy - s / 2))
    canvas = np.zeros((s, s, 3), img.dtype)
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x0 + s, w), min(y0 + s, h)
    if sx1 > sx0 and sy1 > sy0:
        canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    return cv2.resize(canvas, (out_size, out_size)), x0, y0, s


class MediaPipeDetector:
    """Pose (single person, tracked frame-to-frame) + up to ``max_hands`` hands.

    Pose runs on every frame - also while following is stopped - so recordings
    contain the complete MediaPipe output. The follow logic ignores it while stopped.
    """

    def __init__(self, model_complexity=1, pose_detection_conf=0.5, pose_tracking_conf=0.5,
                 max_hands=4, hand_detection_conf=0.6, hand_tracking_conf=0.5,
                 recover_hands=True, roi_detection_conf=0.5):
        self._pose = mp_pose.Pose(model_complexity=model_complexity,
                                  min_detection_confidence=pose_detection_conf,
                                  min_tracking_confidence=pose_tracking_conf)
        # max_hands > 2: MediaPipe stops looking for new hands once it tracks
        # max_hands, so with 2 one person's hands would hide everyone else's gesture.
        self._hands = mp_hands.Hands(max_num_hands=max_hands,
                                     min_detection_confidence=hand_detection_conf,
                                     min_tracking_confidence=hand_tracking_conf)
        self._roi_hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1,
                                         min_detection_confidence=roi_detection_conf) \
            if recover_hands else None
        self._target: Optional[ResumeTarget] = None
        self._target_frames = 0

    # ---- public ------------------------------------------------------------------
    def process(self, frame_bgr) -> Detection:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        hand_res = self._hands.process(rgb)
        hands = list(hand_res.multi_hand_landmarks or [])

        lock_region = self._target.region if self._target else None
        pose = self._pose.process(self._masked(rgb) if self._target else rgb).pose_landmarks
        if self._target is not None:
            pose = self._check_lock(pose)

        recovered = self._recover_hands(rgb, pose.landmark, hands) if pose is not None else []
        return Detection(pose_landmarks=pose, hand_landmarks=hands + recovered,
                         recovered_hands=len(recovered), lock_region=lock_region)

    def reset_pose(self):
        """Forget the tracked person; the next frame detects whoever is most prominent."""
        self._pose.reset()

    def lock_on(self, target: Optional[ResumeTarget]):
        """Make the next pose detection pick the person in ``target.region`` (None = anyone)."""
        self._target, self._target_frames = target, 0
        self._pose.reset()

    def close(self):
        self._pose.close()
        self._hands.close()
        if self._roi_hands is not None:
            self._roi_hands.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ---- locking onto the gesturer -------------------------------------------------
    def _masked(self, rgb):
        h, w = rgb.shape[:2]
        r = self._target.region
        x0, y0 = int(r.x0 * w), int(r.y0 * h)
        x1, y1 = int(math.ceil(r.x1 * w)), int(math.ceil(r.y1 * h))
        out = np.full_like(rgb, MASK_VALUE)
        out[y0:y1, x0:x1] = rgb[y0:y1, x0:x1]
        return out

    def _check_lock(self, pose):
        self._target_frames += 1
        if pose is not None and pose_matches_target(pose.landmark, self._target):
            self._target = None            # locked on; MediaPipe tracking takes over
            return pose
        if pose is not None:
            self._pose.reset()             # found someone else - try again
        if self._target_frames >= LOCK_TIMEOUT_FRAMES:
            self._target = None            # give up; follow whoever is found next
        return None

    # ---- pose-guided hand recovery -------------------------------------------------
    def _recover_hands(self, rgb, pose_lm, hands) -> List[object]:
        if self._roi_hands is None:
            return []
        h, w = rgb.shape[:2]
        found = []
        for side in (0, 1):
            roi = hand_roi(pose_lm, side, w, h)
            if roi is None:
                continue
            cx, cy, size = roi
            wrist = pose_lm[_POSE_HANDS[side][0]]
            wx, wy = wrist.x * w, wrist.y * h
            known = [(hl.landmark[0].x * w, hl.landmark[0].y * h) for hl in hands + found]
            if any(math.hypot(x - wx, y - wy) < 0.35 * size for x, y in known):
                continue                   # MediaPipe already found this hand
            crop, x0, y0, s = crop_square(rgb, cx, cy, size)
            res = self._roi_hands.process(crop)
            if not res.multi_hand_landmarks:
                continue
            lm = res.multi_hand_landmarks[0].landmark
            hand = landmark_pb2.NormalizedLandmarkList(landmark=[
                landmark_pb2.NormalizedLandmark(x=(x0 + p.x * s) / w, y=(y0 + p.y * s) / h,
                                                z=p.z * s / w) for p in lm])
            hx, hy = hand.landmark[0].x * w, hand.landmark[0].y * h
            if math.hypot(hx - wx, hy - wy) > 0.5 * size:
                continue                   # found a different hand in the crop
            if any(math.hypot(x - hx, y - hy) < 0.25 * size for x, y in known):
                continue                   # duplicate of a known hand
            found.append(hand)
        return found


class NoDetector:
    """Used when replaying recordings: the landmarks are already in the file."""

    def reset_pose(self):
        pass

    def lock_on(self, target):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass
