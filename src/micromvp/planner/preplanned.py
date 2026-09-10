"""Prepare a complete nominal DRVG route before hardware execution."""
from dataclasses import dataclass
import math
import threading
import time

from .dynamic_rvg import DynamicRVGPlan


@dataclass(frozen=True)
class FrozenConfiguration:
    x: float
    y: float
    theta: float

    def getX(self): return self.x
    def getY(self): return self.y
    def getTheta(self): return self.theta


def freeze_plan(plan):
    return DynamicRVGPlan(
        status=str(plan.status).split(".")[-1],
        configurations=[FrozenConfiguration(v.getX(),v.getY(),v.getTheta())
                        for v in plan.configurations],
        controller_path=[tuple(point) for point in plan.controller_path],
        temporary_goal=tuple(plan.temporary_goal) if plan.temporary_goal is not None else None,
        final_segment=bool(plan.final_segment),
    )


def prepare_route(planner, start, goal, *, max_steps, timeout,
                  cancelled=lambda: False, progress=lambda *args: None):
    """Advance only through exact planned endpoints; never use measured drift."""
    started = time.monotonic()
    planner.initialize(start,goal)
    plans = []
    for index in range(max_steps):
        if cancelled():
            raise RuntimeError("Preplanning cancelled")
        if time.monotonic()-started > timeout:
            raise RuntimeError(f"Preplanning exceeded {timeout:g} seconds")
        tick = time.perf_counter()
        plan = freeze_plan(planner.step())
        elapsed = time.perf_counter()-tick
        if cancelled():
            raise RuntimeError("Preplanning cancelled")
        if time.monotonic()-started > timeout:
            raise RuntimeError(f"Preplanning exceeded {timeout:g} seconds")
        progress(index+1,plan.status,elapsed)
        if plan.status == "GoalReached":
            if plans:
                plans[-1].final_segment = True
            return plans
        if plan.status not in {"GoalPathAvailable","TemporaryGoalPathAvailable"} or len(plan.configurations)<2:
            raise RuntimeError(f"Preplanning step {index+1}: {plan.status}")
        plans.append(plan)
        if plan.final_segment:
            return plans
        endpoint=plan.configurations[-1]
        if not planner.complete_current_segment():
            raise RuntimeError("Preplanner could not acknowledge its temporary segment")
        planner.update_pose((endpoint.x,endpoint.y,math.degrees(endpoint.theta)%360))
    raise RuntimeError(f"Preplanning reached its {max_steps} step limit")


class PreplanTask:
    """One cancellable background job with no camera or motor access."""

    def __init__(self, factory, start, goal, max_steps, timeout):
        self._cancel = threading.Event()
        self._lock = threading.Lock()
        self._done = False
        self._plans = None
        self._error = None
        self._progress = "Preparing DRVG route..."
        self._thread = threading.Thread(
            target=self._run,args=(factory,start,goal,max_steps,timeout),
            name="drvg-preplan",daemon=True,
        )

    def start(self):
        self._thread.start()

    def _run(self, factory, start, goal, max_steps, timeout):
        try:
            plans=prepare_route(
                factory(),start,goal,max_steps=max_steps,timeout=timeout,
                cancelled=self._cancel.is_set,progress=self._report,
            )
            with self._lock:
                self._plans=plans
        except Exception as error:
            with self._lock:
                self._error=str(error)
        finally:
            with self._lock:
                self._done=True

    def _report(self,index,status,elapsed):
        message=f"Preplanning step {index}: {status} ({elapsed*1000:.1f} ms)"
        print(f"[Preplan] {message}",flush=True)
        with self._lock:
            self._progress=message

    def poll(self):
        with self._lock:
            return self._done,self._plans,self._error,self._progress

    def cancel(self):
        self._cancel.set()

    def join(self):
        self._thread.join()
