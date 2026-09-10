import csv
from types import SimpleNamespace
from micromvp.utils.motion_log import MotionLog
from micromvp.core.models import Action,CarState,RobotObservation


def test_heading_unwrap_and_tracking_gap(tmp_path):
    log=MotionLog(tmp_path,1,{})
    executor=SimpleNamespace(done=True,index=0,motions=[])
    state=CarState(car_id=3)
    for i,heading in enumerate([350,359,2,10]):
        log.record(RobotObservation(3,1,2,heading,timestamp=1+i*.1),
                   Action(-.12,.12),1,executor,state)
    log.record(None,Action.stop(),1,executor,state,"tracking_lost")
    log.record(RobotObservation(3,1,2,180,timestamp=2),Action.stop(),1,executor,state)
    log.close("done")
    rows=list(csv.DictReader(log.path.open()))
    assert [float(r["heading_unwrapped_deg"]) for r in rows[:4]]==[350,359,362,370]
    assert rows[4]["heading_deg"]==""
    assert rows[5]["tracking_epoch"]=="2"
    assert float(rows[5]["heading_unwrapped_deg"])==180
    assert float(rows[0]["left_command"])==-.12


def test_metadata_references_route(tmp_path):
    import json
    log=MotionLog(tmp_path,2,{"goal":[1,2,3]})
    log.set_route_path(tmp_path/"planned.json")
    log.close("cancelled")
    metadata=json.loads(log.path.with_suffix(".json").read_text())
    assert metadata["goal"]==[1,2,3]
    assert metadata["end_reason"]=="cancelled"
    assert metadata["preplanned_route"].endswith("planned.json")
