"""Draws everything onto a frame: the header (the gestures available right now
and what each will do), the status and banner, the pose diagram (unless hidden),
hand-shape labels, the hold-progress bar / "Lower your hands" prompt, and the
lock-on box after a resume."""
import cv2
import mediapipe as mp

from follower.logic import DEFAULT_SETTINGS, FrameResult, GesturePhase, Settings, Status, pair_name

from .detector import Detection

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

POSE_STYLE = mp_draw.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=3)
CONN_STYLE = mp_draw.DrawingSpec(color=(255, 255, 0), thickness=3)
HAND_STYLE = mp_draw.DrawingSpec(color=(200, 200, 200), thickness=2, circle_radius=2)
HAND_CONN_STYLE = mp_draw.DrawingSpec(color=(150, 150, 150), thickness=2)

# BGR colours
GREEN, RED, YELLOW, ORANGE, WHITE = (0, 255, 0), (0, 0, 255), (0, 255, 255), (0, 200, 255), \
    (255, 255, 255)

STATUS_TEXT = {
    Status.FOLLOWING: ("FOLLOWING", GREEN),
    Status.SEARCHING: ("FOLLOWING (searching for person...)", GREEN),
    Status.STOPPED: ("STOPPED", RED),
}

HEADER_COLOR = WHITE
HEADER_Y0, HEADER_LINE_H = 20, 20


def header_lines(result: FrameResult):
    """What the person can currently do, and what each action will cause -
    e.g. 'Hold open palm + fist to STOP tracking'."""
    lines = []
    for opt in result.options:
        pair = pair_name(opt.pair)
        if opt.action == "stop":
            effect = "STOP tracking"
        elif opt.action == "resume":
            effect = "RESUME tracking (on whoever shows it)"
        else:
            effect = ("HIDE" if result.show_skeleton else "SHOW") + " the body pose drawing"
        lines.append(f"Hold {pair} to {effect}")
    return lines


def draw_header(img, result: FrameResult):
    for i, line in enumerate(header_lines(result)):
        draw_label(img, line, HEADER_COLOR, y=HEADER_Y0 + i * HEADER_LINE_H, scale=0.52)
    return HEADER_Y0 + len(result.options) * HEADER_LINE_H + 14  # y the next element can use


def draw_label(img, text, color, y=30, x=12, scale=0.8):
    """Outlined text, shrunk if needed so it fits inside the frame."""
    room = img.shape[1] - x - 8
    width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0][0]
    if width > room > 0:
        scale *= room / width
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def draw_pose(img, pose_landmarks):
    """Mark the pose points and connect them into the stick-figure diagram."""
    mp_draw.draw_landmarks(img, pose_landmarks, mp_pose.POSE_CONNECTIONS, POSE_STYLE, CONN_STYLE)


def draw_hands(img, detection: Detection):
    """Hand skeletons (used for replays, where there is no video to show the hands)."""
    for hand in detection.hand_landmarks:
        mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS, HAND_STYLE, HAND_CONN_STYLE)


def draw_lock_region(img, detection: Detection):
    """Dashed box where we're looking for the person who made the resume gesture."""
    r = detection.lock_region
    if r is None:
        return
    h, w = img.shape[:2]
    x0, y0, x1, y1 = int(r.x0 * w), int(r.y0 * h), int(r.x1 * w) - 1, int(r.y1 * h) - 1
    for a, b in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)),
                 ((x0, y1), (x0, y0))):
        n = max(int(max(abs(b[0] - a[0]), abs(b[1] - a[1])) / 12), 1)
        for i in range(0, n, 2):
            p = (a[0] + (b[0] - a[0]) * i // n, a[1] + (b[1] - a[1]) * i // n)
            q = (a[0] + (b[0] - a[0]) * (i + 1) // n, a[1] + (b[1] - a[1]) * (i + 1) // n)
            cv2.line(img, p, q, YELLOW, 2)
    draw_label(img, "locking on...", YELLOW, y=max(y0 + 22, 22), x=x0 + 6, scale=0.55)


def draw_hand_labels(img, detection: Detection, result: FrameResult):
    """Write the recognised gesture under the wrist of each hand that counted."""
    h, w = img.shape[:2]
    for idx, gesture in result.hand_gestures:
        p = detection.hand_landmarks[idx].landmark[0]
        cv2.putText(img, gesture.name, (int(p.x * w) - 20, int(p.y * h) + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, ORANGE, 2)


ACTION_VERB = {"stop": "stop", "resume": "resume", "toggle_drawing": "toggle the pose drawing"}


def draw_gesture_progress(img, result: FrameResult, settings: Settings):
    """Hold-progress bar for whichever gesture is currently mid-hold; prompt to
    lower the hands while one has just fired and is waiting to be re-armed."""
    h = img.shape[0]
    active = next((o for o in result.options
                  if o.phase in (GesturePhase.HOLDING, GesturePhase.WAIT_RELEASE)), None)
    if active is None:
        return
    if active.phase is GesturePhase.HOLDING:
        frac = min(active.hold_elapsed / settings.hold_seconds, 1.0)
        draw_label(img, f"HOLD {pair_name(active.pair)} to {ACTION_VERB[active.action]}: "
                        f"{active.hold_elapsed:.1f} / {settings.hold_seconds:g} s",
                   ORANGE, y=h - 40, scale=0.65)
        x0, x1, y0 = 12, 12 + int(300 * frac), h - 28
        cv2.rectangle(img, (12, y0), (312, y0 + 14), WHITE, 1)
        if x1 > x0:
            cv2.rectangle(img, (x0, y0), (x1, y0 + 14), ORANGE, -1)
    else:
        draw_label(img, "Lower your hands", WHITE, y=h - 20, scale=0.65)


def render(img, detection: Detection, result: FrameResult,
           settings: Settings = DEFAULT_SETTINGS, show_hands: bool = False):
    """Draw everything for one frame (in place)."""
    if show_hands:
        draw_hands(img, detection)
    draw_lock_region(img, detection)
    if result.status is Status.FOLLOWING and result.show_skeleton and detection.pose_landmarks:
        draw_pose(img, detection.pose_landmarks)   # only the person actually followed
    draw_hand_labels(img, detection, result)
    y = draw_header(img, result)
    text, color = STATUS_TEXT[result.status]
    draw_label(img, text, color, y=y)
    if result.flash_text:
        draw_label(img, result.flash_text, YELLOW, y=y + 35)
    draw_gesture_progress(img, result, settings)
    return img
