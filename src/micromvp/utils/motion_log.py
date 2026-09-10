"""Measured pose and command history for a single hardware run."""
import csv
import json
import time
from pathlib import Path


class MotionLog:
    FIELDS = (
        "elapsed_s", "observation_timestamp", "tracking_epoch", "robot_id",
        "x_cm", "y_cm", "heading_deg", "heading_unwrapped_deg",
        "left_command", "right_command", "segment", "motion_index", "motion",
        "reverse", "target_x_cm", "target_y_cm", "target_heading_deg",
        "controller_state", "heading_error_deg", "event",
    )

    def __init__(self, directory, run_number, metadata):
        self.path = Path(directory) / f"measured_run_{run_number:03d}_{time.time_ns()}.csv"
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self._file = self.path.open("x",newline="",encoding="utf-8")
        self._writer = csv.DictWriter(self._file,fieldnames=self.FIELDS)
        self._writer.writeheader()
        self._started = time.monotonic()
        self._last_flush = self._started
        self._previous = None
        self._epoch = 0
        self._unwrapped = 0.0
        self.metadata = dict(metadata,started_at=time.time(),csv=str(self.path.resolve()))
        self._save_metadata()
        print(f"[MotionLog] {self.path.resolve()}",flush=True)

    def _save_metadata(self):
        self.path.with_suffix(".json").write_text(json.dumps(self.metadata,indent=2))

    def set_route_path(self,path):
        self.metadata["preplanned_route"] = str(Path(path).resolve())
        self._save_metadata()

    def record(self,observation,action,segment,executor,state,event="sample"):
        row={key:"" for key in self.FIELDS}
        now=time.monotonic()
        row.update(elapsed_s=now-self._started,left_command=action.left_speed,
                   right_command=action.right_speed,segment=segment,
                   controller_state=state.status_label,event=event)
        if observation is None:
            self._previous=None
        else:
            heading=observation.theta%360
            timestamp=observation.timestamp
            if self._previous is None or not 0<=timestamp-self._previous[0]<=.5:
                self._epoch+=1
                self._unwrapped=heading
            else:
                self._unwrapped+=(heading-self._previous[1]+180)%360-180
            self._previous=(timestamp,heading)
            row.update(observation_timestamp=timestamp,tracking_epoch=self._epoch,
                       robot_id=observation.robot_id,x_cm=observation.x,y_cm=observation.y,
                       heading_deg=heading,heading_unwrapped_deg=self._unwrapped)
        if not executor.done and executor.index<len(executor.motions):
            motion=executor.motions[executor.index]
            row.update(motion_index=executor.index+1,motion=motion.kind,
                       reverse=motion.reverse,target_x_cm=motion.end[0],
                       target_y_cm=motion.end[1],target_heading_deg=motion.end[2])
        metadata=getattr(state,"metadata",{})
        row["heading_error_deg"]=metadata.get("rotation_error_deg","")
        self._writer.writerow(row)
        if now-self._last_flush>=1:
            self._file.flush()
            self._last_flush=now

    def close(self,reason):
        self._file.close()
        self.metadata.update(ended_at=time.time(),end_reason=reason)
        self._save_metadata()
