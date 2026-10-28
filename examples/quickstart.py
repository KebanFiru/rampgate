"""README quickstart: three ramping jobs on one GPU in the simulator (`pip install rampgate`)."""

from rampgate import Device, FifoPolicy, Job, MarkStable, Resources, Start
from rampgate.sim import RampPlateau, Scenario, SimJob, simulate

gpu = Device("gpu0", Resources(vram=24))  # GB here; any unit works
ramp = RampPlateau(load_s=20, ramp_s=40, plateau=Resources(vram=9), duration_s=300)
jobs = tuple(SimJob(Job(f"train-{i}", {"gpu0": None}), ramp, submit_at=0) for i in range(3))

trace = simulate(Scenario((gpu,), jobs), FifoPolicy(), max_time=3000)
for tick in trace.ticks:
    for action in tick.decision.actions:
        if isinstance(action, Start | MarkStable):
            print(f"t={tick.now:5.0f}  {type(action).__name__:10} {action.job_id}")
print("OOM kills:", trace.world.oom_count, " finished at t =", trace.ticks[-1].now)
