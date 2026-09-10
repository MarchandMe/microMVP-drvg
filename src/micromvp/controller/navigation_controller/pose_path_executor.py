"""Execute an SE(2) path as stopped turns and individual straight segments."""
from dataclasses import dataclass
import math
import time

from micromvp.core.models import Action


def heading_delta(start, end):
    return (end - start + 180.0) % 360.0 - 180.0


@dataclass(frozen=True)
class Motion:
    kind: str
    start: tuple[float, float, float]
    end: tuple[float, float, float]
    reverse: bool = False


def differential_drive_motions(poses, *, orientation_invariant=False):
    """Replace holonomic translations with validated turn/drive/turn motions."""
    if not poses:
        return []
    motions = []
    current = tuple(poses[0])
    for index, target in enumerate(poses[1:], start=1):
        x, y, heading = target
        preserve_heading = not orientation_invariant or index == len(poses) - 1
        distance = math.hypot(x - current[0], y - current[1])
        if distance > 1e-6:
            travel = math.degrees(math.atan2(y - current[1], x - current[0])) % 360
            backward = (travel + 180.0) % 360.0
            forward_turn = abs(heading_delta(current[2], travel)) + (abs(heading_delta(travel, heading)) if preserve_heading else 0.0)
            reverse_turn = abs(heading_delta(current[2], backward)) + (abs(heading_delta(backward, heading)) if preserve_heading else 0.0)
            reverse = reverse_turn + 1e-6 < forward_turn
            if reverse:
                travel = backward
            if abs(heading_delta(current[2], travel)) > 1e-6:
                aligned = (current[0], current[1], travel)
                motions.append(Motion("rotate", current, aligned))
                current = aligned
            destination = (x, y, travel)
            motions.append(Motion("drive", current, destination, reverse=reverse))
            current = destination
        if preserve_heading and abs(heading_delta(current[2], heading)) > 1e-6:
            destination = (x, y, heading % 360)
            motions.append(Motion("rotate", current, destination))
            current = destination
    return motions


class PosePathExecutor:
    """Never feed a turn marker or a multi-corner path to pure pursuit."""

    def __init__(self, controller, motion_is_valid, reject_blocked_motion=True, *, orientation_invariant=False):
        self.controller = controller
        self.motion_is_valid = motion_is_valid
        self.reject_blocked_motion = reject_blocked_motion
        self.orientation_invariant = orientation_invariant
        self.motions = []
        self.index = 0
        self.started = False
        self.done = True
        self._phase_started = 0.0
        self._last_rotation_log = 0.0

    def reset(self):
        self.motions = []
        self.index = 0
        self.started = False
        self.done = True

    def load(self, configurations):
        poses = [(v.getX(), v.getY(), math.degrees(v.getTheta()) % 360)
                 for v in configurations]
        motions = []
        for motion in differential_drive_motions(poses, orientation_invariant=self.orientation_invariant):
            merge = False
            if (motions and motions[-1].kind == motion.kind == "rotate"
                    and math.dist(motions[-1].start[:2], motion.start[:2]) < 1e-6):
                previous = motions[-1]
                first_delta = heading_delta(previous.start[2],previous.end[2])
                next_delta = heading_delta(motion.start[2],motion.end[2])
                combined_delta = heading_delta(previous.start[2],motion.end[2])
                # Combining the identical signed arc only removes a pause.
                # It needs no new clearance test, including across 0 degrees.
                same_arc = (
                    first_delta * next_delta >= 0
                    and abs(first_delta + next_delta - combined_delta) < 1e-6
                )
                merge = same_arc or self.motion_is_valid(previous.start,motion.end)
            if merge:
                motions[-1] = Motion("rotate",motions[-1].start,motion.end)
            else:
                motions.append(motion)
        for motion in motions:
            if self.reject_blocked_motion and not self.motion_is_valid(motion.start, motion.end):
                raise ValueError(
                    f"Unsafe {motion.kind} near ({motion.start[0]:.2f}, "
                    f"{motion.start[1]:.2f}); inflated swept footprint is blocked"
                )
        self.motions = motions
        self.index = 0
        self.started = False
        self.done = not motions

    def step(self, observation):
        if self.done:
            return Action.stop()
        motion = self.motions[self.index]
        # Recheck the remaining sweep at the measured pose, including drift
        # accumulated during a physical turn. Never substitute a planned pose.
        target = (
            (observation.x, observation.y, motion.end[2])
            if motion.kind == "rotate" else motion.end
        )
        if self.reject_blocked_motion and not self.motion_is_valid(observation.pose, target):
            self.controller.reset()
            raise ValueError(f"Measured {motion.kind} sweep lost obstacle clearance")
        if not self.started:
            self._phase_started = time.monotonic()
            self._last_rotation_log = self._phase_started
            print(
                f"[Motion] {self.index + 1}/{len(self.motions)} {motion.kind}"
                f"{' reverse' if motion.reverse else ''}: "
                f"measured=({observation.x:.2f}, {observation.y:.2f}, {observation.theta:.1f}°) "
                f"target=({motion.end[0]:.2f}, {motion.end[1]:.2f}, {motion.end[2]:.1f}°)",
                flush=True,
            )
            if motion.kind == "rotate":
                print(
                    f"[Motion] signed turn={heading_delta(observation.theta,motion.end[2]):+.1f}°",
                    flush=True,
                )
                self.controller.rotate_to(motion.end[2])
            else:
                dx,dy=motion.end[0]-observation.x,motion.end[1]-observation.y
                bearing=(math.degrees(math.atan2(dy,dx))+(180 if motion.reverse else 0))%360
                print(
                    f"[Motion] measured-position drive bearing={bearing:.1f}°; "
                    f"queued bearing={motion.end[2]:.1f}°; "
                    f"remaining={math.hypot(dx,dy):.2f} cm",
                    flush=True,
                )
                path = [observation.position, motion.end[:2]]
                if motion.reverse:
                    self.controller.set_path(path, reverse=True)
                else:
                    self.controller.set_path(path)
            self.started = True
        action = self.controller.step(observation)
        # Match navigation.py: enter a following path once the pre-turn
        # reaches tolerance. Final/standalone rotations still require settling.
        next_is_drive = (
            self.index + 1 < len(self.motions)
            and self.motions[self.index + 1].kind == "drive"
        )
        complete = (
            self.controller.car_state.status_label == "ROTATION_DONE"
            or (next_is_drive and self.controller.car_state.status_label == "ROTATION_STABLE")
            if motion.kind == "rotate" else
            self.controller.car_state.status_label == "FINISHED"
        )
        if complete:
            print(
                f"[Motion] {motion.kind} complete after "
                f"{time.monotonic() - self._phase_started:.2f}s; "
                f"measured heading={observation.theta:.1f}°",
                flush=True,
            )
            self.index += 1
            self.started = False
            self.done = self.index == len(self.motions)
            return Action.stop()
        if motion.kind == "rotate" and time.monotonic() - self._last_rotation_log >= 1.0:
            self._last_rotation_log = time.monotonic()
            print(
                f"[Motion] {self.status}; "
                f"L={action.left_speed:.3f} R={action.right_speed:.3f}",
                flush=True,
            )
        return action

    @property
    def status(self):
        if self.done:
            return "motion sequence complete"
        motion = self.motions[self.index]
        if motion.kind == "rotate":
            metadata = getattr(self.controller.car_state, "metadata", {})
            error = metadata.get("rotation_error_deg")
            detail = f"; error {error:+.1f}°" if error is not None else ""
            if self.controller.car_state.status_label == "ROTATION_STABLE":
                detail += "; settling"
            return f"turn in place to {motion.end[2]:.1f}°{detail}"
        return f"{'reverse' if motion.reverse else 'drive'} to ({motion.end[0]:.2f}, {motion.end[1]:.2f})"
