"""Turns the per-frame results into a timeline of the major actions, and
formats it as the human-readable ``*.actions.txt`` report (pure Python)."""
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from .config import DEFAULT_SETTINGS, Settings
from .gestures import GesturePhase, gesture_pair, pair_name
from .tracking import FrameResult, Status

ACTIONS = ("stop", "resume", "toggle_drawing")
VERB = {"stop": "stop following", "resume": "resume following"}


class ActionKind(Enum):
    START = "START"
    PERSON_FOUND = "PERSON_FOUND"
    PERSON_LOST = "PERSON_LOST"
    GESTURE_STARTED = "GESTURE_STARTED"
    GESTURE_TOO_SHORT = "GESTURE_TOO_SHORT"
    STOPPED = "FOLLOWING_STOPPED"
    HANDS_LOWERED = "HANDS_LOWERED"
    RESUMED = "FOLLOWING_RESUMED"
    DRAWING_SHOWN = "POSE_DIAGRAM_SHOWN"
    DRAWING_HIDDEN = "POSE_DIAGRAM_HIDDEN"
    END = "END"


@dataclass(frozen=True)
class Action:
    t: float
    kind: ActionKind
    message: str


class ActionLog:
    """Feed it every FrameResult (``observe``), then call ``finish``.

    Person found/lost events are debounced by ``settings.event_debounce`` so a
    single-frame detector dropout isn't reported; they are time-stamped with
    the moment the change actually began.

    The stop / resume / toggle-drawing gestures are tracked independently (each
    is only listenable in some states - see ``FrameResult.options``); a gesture
    that stops being listenable is reset without logging anything, so switching
    states never fabricates a "hands lowered" event.
    """

    def __init__(self, settings: Settings = DEFAULT_SETTINGS):
        self.settings = settings
        self._actions: List[Action] = []
        self._phase = {a: GesturePhase.ARMED for a in ACTIONS}
        self._hold_elapsed = {a: 0.0 for a in ACTIONS}
        self._present: Optional[bool] = None      # reported presence (None = not following)
        self._candidate: Optional[bool] = None    # presence change waiting for debounce
        self._candidate_since = 0.0
        self._last_t: Optional[float] = None
        self.start_t: Optional[float] = None
        self.frames = 0
        self.following_frames = 0                 # status FOLLOWING (person in view)
        self.stopped_frames = 0
        self.finished = False

    # ---- public ----------------------------------------------------------------
    @property
    def actions(self) -> List[Action]:
        return sorted(self._actions, key=lambda a: a.t)  # stable: same-time order kept

    def kinds(self) -> List[ActionKind]:
        return [a.kind for a in self.actions]

    def observe(self, r: FrameResult, now: float):
        if self.start_t is None:
            self.start_t = now
            self._add(now, ActionKind.START, "Session started - following the first person "
                                             "who comes into view")
        self.frames += 1
        self.following_frames += r.status is Status.FOLLOWING
        self.stopped_frames += r.status is Status.STOPPED
        # A stop/resume is logged from its explicit flag, not the phase transition below:
        # by the time we see this FrameResult, ``following`` (and so the option list) has
        # already flipped, so the option that just fired may no longer be listed.
        self._emit_fired(r, now)
        self._observe_gestures(r, now)
        self._observe_presence(r, now)
        self._last_t = now

    def finish(self, now: Optional[float] = None):
        if self.finished:
            return
        now = self._last_t if now is None else now
        if now is None:          # no frames at all
            now = 0.0
            self.start_t = 0.0
        if self._candidate is not None and now - self._candidate_since >= self.settings.event_debounce:
            self._report_presence(self._candidate, self._candidate_since)
        counts = self._counts()
        self._add(now, ActionKind.END, f"Session ended - {counts['stops']} stop(s), "
                                       f"{counts['resumes']} resume(s), "
                                       f"{counts['drawing_toggles']} drawing toggle(s)")
        self.finished = True

    def summary(self) -> Dict[str, float]:
        duration = (self._last_t - self.start_t) if self.frames else 0.0
        return {
            "frames": self.frames,
            "duration_s": round(duration, 3),
            "followed_fraction": round(self.following_frames / self.frames, 3) if self.frames else 0.0,
            "stopped_fraction": round(self.stopped_frames / self.frames, 3) if self.frames else 0.0,
            **self._counts(),
        }

    def _counts(self):
        return {"stops": sum(a.kind is ActionKind.STOPPED for a in self._actions),
                "resumes": sum(a.kind is ActionKind.RESUMED for a in self._actions),
                "drawing_toggles": sum(a.kind in (ActionKind.DRAWING_SHOWN,
                                                  ActionKind.DRAWING_HIDDEN)
                                       for a in self._actions)}

    # ---- internals ---------------------------------------------------------------
    def _add(self, t, kind, message):
        self._actions.append(Action(t, kind, message))

    def _emit_fired(self, r: FrameResult, now):
        s = self.settings
        if r.toggled_stop:
            pair = pair_name(gesture_pair(s, "stop_gesture"))
            self._add(now, ActionKind.STOPPED,
                      f"FOLLOWING STOPPED - {pair} held for {s.hold_seconds:g} s")
        if r.toggled_resume:
            pair = pair_name(gesture_pair(s, "resume_gesture"))
            self._add(now, ActionKind.RESUMED,
                      f"FOLLOWING RESUMED - {pair} held for {s.hold_seconds:g} s; "
                      "locking onto the person who made the gesture")
        if r.toggled_draw:
            pair = pair_name(gesture_pair(s, "draw_toggle_gesture"))
            shown = r.show_skeleton
            self._add(now, ActionKind.DRAWING_SHOWN if shown else ActionKind.DRAWING_HIDDEN,
                      f"POSE DIAGRAM {'SHOWN' if shown else 'HIDDEN'} - {pair} held for "
                      f"{s.hold_seconds:g} s (tracking keeps running)")

    def _observe_gestures(self, r: FrameResult, now):
        options = {o.action: o for o in r.options}
        for action in ACTIONS:
            if action in options:
                self._observe_one_gesture(action, options[action], r.show_skeleton, now)
            else:                # not currently listenable: reset silently, no event
                self._phase[action], self._hold_elapsed[action] = GesturePhase.ARMED, 0.0

    def _observe_one_gesture(self, action, opt, show_skeleton, now):
        """STARTED / TOO_SHORT / HANDS_LOWERED only - firing itself is handled by
        ``_emit_fired`` (by the time we get here, a fired stop/resume's option may
        already be gone, because ``following`` has flipped)."""
        s, prev = self.settings, self._phase[action]
        pair = pair_name(opt.pair)
        if prev is GesturePhase.ARMED and opt.phase is GesturePhase.HOLDING:
            if action == "toggle_drawing":
                goal = "hide the pose diagram" if show_skeleton else "show the pose diagram"
            else:
                goal = VERB[action]
            self._add(now, ActionKind.GESTURE_STARTED,
                      f"{pair.capitalize()} detected - holding to {goal} (needs "
                      f"{s.hold_seconds:g} s)")
        elif prev is GesturePhase.HOLDING and opt.phase is GesturePhase.ARMED:
            self._add(now, ActionKind.GESTURE_TOO_SHORT,
                      f"{pair.capitalize()} released after {self._hold_elapsed[action]:.1f} s - "
                      f"too short, nothing changed")
        elif prev is GesturePhase.WAIT_RELEASE and opt.phase is not GesturePhase.WAIT_RELEASE:
            self._add(now, ActionKind.HANDS_LOWERED,
                      f"Hands lowered - the {pair} gesture is accepted again")
        self._phase[action], self._hold_elapsed[action] = opt.phase, opt.hold_elapsed

    def _observe_presence(self, r: FrameResult, now):
        if not r.following:                       # nobody is followed while stopped
            self._present, self._candidate = None, None
            return
        if self._present is None:                 # session start or just resumed
            self._present = False
        seen = r.status is Status.FOLLOWING
        if seen == self._present:
            self._candidate = None
        elif self._candidate is not seen:
            self._candidate, self._candidate_since = seen, now
        elif now - self._candidate_since >= self.settings.event_debounce:
            self._report_presence(seen, self._candidate_since)

    def _report_presence(self, seen, t):
        if seen:
            self._add(t, ActionKind.PERSON_FOUND, "Person in view - following them")
        else:
            self._add(t, ActionKind.PERSON_LOST, "Followed person no longer detected - searching")
        self._present, self._candidate = seen, None


def _fmt_t(t):
    m, s = divmod(max(t, 0.0), 60)
    return f"{int(m):02d}:{s:06.3f}"


def format_report(log: ActionLog, source: Dict[str, object]) -> str:
    """Render the action log as the text written to ``<name>.actions.txt``.

    ``source`` is shown as a header (e.g. name, kind, resolution, fps).
    """
    log.finish()
    s = log.settings
    start = log.start_t or 0.0
    stop_name = pair_name(gesture_pair(s, "stop_gesture"))
    resume_name = pair_name(gesture_pair(s, "resume_gesture"))
    draw_name = pair_name(gesture_pair(s, "draw_toggle_gesture"))
    lines = ["pose_follower - action log", "=" * 60]
    lines += [f"{k:<12}{v}" for k, v in source.items()]
    lines.append(f"{'rules':<12}stop = {stop_name}; resume = {resume_name}; "
                 f"toggle pose diagram = {draw_name}; each held {s.hold_seconds:g} s; "
                 f"hands down for {s.release_seconds:g} s re-arms a gesture")
    lines += ["", f"{'TIME':<10} {'ACTION':<19} DETAILS", "-" * 60]
    for a in log.actions:
        lines.append(f"{_fmt_t(a.t - start):<10} {a.kind.value:<19} {a.message}")
    sm = log.summary()
    lines += ["", "Summary", "-" * 60,
              f"duration            {sm['duration_s']:.2f} s ({sm['frames']} frames)",
              f"person followed     {100 * sm['followed_fraction']:.0f}% of frames",
              f"stopped             {100 * sm['stopped_fraction']:.0f}% of frames",
              f"stops / resumes     {sm['stops']} / {sm['resumes']}",
              f"drawing toggles     {sm['drawing_toggles']}",
              ""]
    return "\n".join(lines)
