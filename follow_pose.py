"""Webcam pose follower with gesture toggle.

- Locks onto the first person MediaPipe detects and draws their pose "sticks".
- Showing a FIST and an OPEN PALM (in either order, within GESTURE_WINDOW
  seconds, or one in each hand at once) toggles following OFF.
- The next fist + open palm toggles following back ON, and so on.

Keys: q / ESC = quit.
"""
import math
import time
from enum import Enum

import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# ---- tunables --------------------------------------------------------------
CAMERA_INDEX = 0
GESTURE_WINDOW = 3.0     # s: fist and palm must both be seen within this window
STABLE_FRAMES = 4        # consecutive frames a gesture must hold to count
COOLDOWN = 2.0           # s: ignore gestures after a toggle
HAND_NEAR_BODY = 3.0     # max hand-to-person-wrist distance, in shoulder widths
MIN_VISIBILITY = 0.5     # pose landmark visibility needed to draw / use


class Gesture(Enum):
    NONE = 0
    FIST = 1
    PALM = 2


def _d(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def classify_hand(lm):
    """Rotation-invariant fist / open-palm classifier from 21 hand landmarks."""
    wrist = lm[0]
    # (tip, pip) for index, middle, ring, pinky
    fingers = [(8, 6), (12, 10), (16, 14), (20, 18)]
    extended = sum(_d(lm[t], wrist) > _d(lm[p], wrist) * 1.1 for t, p in fingers)
    curled = sum(_d(lm[t], wrist) < _d(lm[p], wrist) for t, p in fingers)
    hand_size = _d(lm[0], lm[9]) or 1e-6
    thumb_out = _d(lm[4], lm[5]) > 0.6 * hand_size  # thumb tip away from index base
    if extended == 4 and thumb_out:
        return Gesture.PALM
    if curled == 4:
        return Gesture.FIST
    return Gesture.NONE


class GestureToggle:
    """Debounces per-frame gestures and fires when a fist + palm pair is seen."""

    def __init__(self):
        self.streak = {Gesture.FIST: 0, Gesture.PALM: 0}
        self.last_seen = {Gesture.FIST: None, Gesture.PALM: None}
        self.blocked_until = 0.0

    def update(self, gestures, now):
        """gestures: list of Gesture seen this frame. Returns True on toggle."""
        for g in self.streak:
            if g in gestures:
                self.streak[g] += 1
                if self.streak[g] >= STABLE_FRAMES:
                    self.last_seen[g] = now
            else:
                self.streak[g] = 0
        if now < self.blocked_until:
            return False
        f, p = self.last_seen[Gesture.FIST], self.last_seen[Gesture.PALM]
        if f is not None and p is not None and abs(f - p) <= GESTURE_WINDOW \
                and now - max(f, p) <= GESTURE_WINDOW:
            self.last_seen = {Gesture.FIST: None, Gesture.PALM: None}
            self.streak = {Gesture.FIST: 0, Gesture.PALM: 0}
            self.blocked_until = now + COOLDOWN
            return True
        return False


def hands_of_person(hand_results, pose_lm, shoulder_w):
    """Gestures from hands that belong to the tracked person.

    If no pose is available (e.g. while stopped) every visible hand counts.
    """
    if not hand_results.multi_hand_landmarks:
        return []
    out = []
    for hl in hand_results.multi_hand_landmarks:
        lm = hl.landmark
        if pose_lm is not None and shoulder_w:
            wrists = [pose_lm[15], pose_lm[16]]
            if min(_d(lm[0], w) for w in wrists) > HAND_NEAR_BODY * shoulder_w:
                continue
        out.append(classify_hand(lm))
    return out


def shoulder_width(pose_lm):
    l, r = pose_lm[11], pose_lm[12]
    if min(l.visibility, r.visibility) < MIN_VISIBILITY:
        return None
    return _d(l, r)


def draw_label(img, text, color, y=30):
    cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {CAMERA_INDEX}")

    following = True
    toggler = GestureToggle()
    flash_until, flash_text = 0.0, ""

    pose = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6,
                           min_tracking_confidence=0.5)
    pose_style = mp_draw.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=3)
    conn_style = mp_draw.DrawingSpec(color=(255, 255, 0), thickness=3)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # mirror view
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            hand_res = hands.process(rgb)
            # Pose only runs while following; this also resets the tracker so the
            # next person to resume is picked up fresh.
            pose_res = pose.process(rgb) if following else None
            rgb.flags.writeable = True

            pose_lm = pose_res.pose_landmarks.landmark if pose_res and pose_res.pose_landmarks else None
            sw = shoulder_width(pose_lm) if pose_lm else None

            now = time.time()
            gestures = hands_of_person(hand_res, pose_lm if following else None, sw)
            if toggler.update(gestures, now):
                following = not following
                flash_text = "FOLLOWING RESUMED" if following else "FOLLOWING STOPPED"
                flash_until = now + 1.5
                if not following:
                    pose.reset() if hasattr(pose, "reset") else None

            if following and pose_res and pose_res.pose_landmarks:
                mp_draw.draw_landmarks(frame, pose_res.pose_landmarks,
                                       mp_pose.POSE_CONNECTIONS, pose_style, conn_style)

            if hand_res.multi_hand_landmarks:
                for hl, g in zip(hand_res.multi_hand_landmarks, gestures if len(gestures) == len(hand_res.multi_hand_landmarks) else []):
                    h, w = frame.shape[:2]
                    p = hl.landmark[0]
                    cv2.putText(frame, g.name, (int(p.x * w) - 20, int(p.y * h) + 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

            if following:
                status = "FOLLOWING" if pose_lm else "FOLLOWING (searching for person...)"
                draw_label(frame, status, (0, 255, 0))
            else:
                draw_label(frame, "STOPPED - show fist + open palm to resume", (0, 0, 255))
            if now < flash_until:
                draw_label(frame, flash_text, (0, 255, 255), y=65)

            cv2.imshow("Pose Follower", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        pose.close()
        hands.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
