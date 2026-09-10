"""Nominal preplanning is isolated from physical execution."""
import math
import pytest
from micromvp.planner.dynamic_rvg import DynamicRVGPlan
from micromvp.planner.preplanned import FrozenConfiguration,PreplanTask,prepare_route

def plan(a,b,final=False):
    v=lambda p:FrozenConfiguration(p[0],p[1],math.radians(p[2]))
    return DynamicRVGPlan("GoalPathAvailable" if final else "TemporaryGoalPathAvailable",
                         [v(a),v(b)],[a[:2],b[:2]],None if final else b[:2],final)

class Planner:
    def __init__(self):
        self.events=[]
        self.plans=[plan((10,10,0),(20,10,0)),plan((20,10,0),(30,10,90),True)]
    def initialize(self,a,b):self.events.append(("initialize",a,b))
    def step(self):
        self.events.append(("step",))
        return self.plans.pop(0)
    def complete_current_segment(self):
        self.events.append(("complete",))
        return True
    def update_pose(self,p):self.events.append(("pose",p))

def test_nominal_endpoints_and_final_heading():
    p=Planner()
    result=prepare_route(p,(10,10,0),(30,10,90),max_steps=10,timeout=5)
    assert len(result)==2 and result[-1].final_segment
    assert result[-1].configurations[-1].theta==pytest.approx(math.pi/2)
    assert p.events==[("initialize",(10,10,0),(30,10,90)),("step",),("complete",),
                      ("pose",(20,10,0)),("step",)]

def test_cancel_returns_no_partial_route():
    p=Planner()
    with pytest.raises(RuntimeError,match="cancelled"):
        prepare_route(p,(10,10,0),(30,10,90),max_steps=10,timeout=5,cancelled=lambda:True)
    assert len(p.events)==1

def test_failure_returns_no_partial_route():
    p=Planner()
    p.plans[1]=DynamicRVGPlan("CurrentPoseInvalid",[],[],None,False)
    with pytest.raises(RuntimeError,match="step 2: CurrentPoseInvalid"):
        prepare_route(p,(10,10,0),(30,10,90),max_steps=10,timeout=5)

def test_background_result():
    t=PreplanTask(Planner,(10,10,0),(30,10,90),10,5)
    t.start();t.join()
    done,plans,error,progress=t.poll()
    assert done and error is None and len(plans)==2
    assert "step 2" in progress

@pytest.mark.rvg
def test_native_route_finishes_before_execution():
    from micromvp.core.models import WorkspaceConfig
    from micromvp.planner.dynamic_rvg import DynamicRVGSession,DynamicRVGSettings
    p=DynamicRVGSession(WorkspaceConfig(100,60,4.2,4.8,2.1,4.5,4.2,10,30,[3]),
        [[(45,15),(49,15),(49,45),(45,45)]],
        DynamicRVGSettings(optimal=True,rotational_weight=5,robot_geometry_scale=1.3))
    route=prepare_route(p,(15,30,0),(85,30,0),max_steps=30,timeout=15)
    assert route and route[-1].final_segment
    end=route[-1].configurations[-1]
    assert (end.x,end.y,math.degrees(end.theta))==pytest.approx((85,30,0))
