# MicroMVP

A multi-robot control framework for small differential-drive cars tracked by
an overhead camera. Robots carry ArUco markers, an overhead camera works out
where they are, and wheel commands go back over an ESP-NOW link.

The same controller code runs against simulation or real hardware, because
control logic never talks to hardware directly — it goes through three
layers you can replace one at a time.

```
┌───────────────────────────────────────────────────────────────────┐
│                               GUI                                 │
│        draws the workspace, takes clicks and drawn paths          │
└───────────────────────────────────────────────────────────────────┘
                                 ↑↓
┌───────────────────────────────────────────────────────────────────┐
│                           Coordinator                             │
│    distributes observations, collects actions, plans paths,       │
│    assigns tasks, bridges the GUI                                 │
└───────────────────────────────────────────────────────────────────┘
                                 ↑↓
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│  Controller 1   │   │  Controller 2   │   │  Controller N   │
│  one per robot  │   │  one per robot  │   │  one per robot  │
│  pose → wheels  │   │  pose → wheels  │   │  pose → wheels  │
└─────────────────┘   └─────────────────┘   └─────────────────┘
                                 ↑↓
┌───────────────────────────────────────────────────────────────────┐
│                           Environment                             │
│    SimEnv or RealEnv — observe() reports poses,                   │
│    apply_actions() drives the wheels                              │
└───────────────────────────────────────────────────────────────────┘
```

The Coordinator is the only layer that touches the Environment: it passes
each robot's observation down to that robot's Controller, and sends the
collected actions back.

---

## Install

```bash
conda create -n micromvp python=3.12
conda activate micromvp

git clone <this repo>
cd micromvp_v4
pip install -e .
```

The RVG path planner is optional and installed separately — see
[Path planning](#path-planning). Without it the system still runs, planning
straight lines instead of routes around obstacles.

---

## Quick start

### 1. Connect the Xiao AP

Plug the AP board into USB. `actuation.serial_port` is `auto` by
default, so there is nothing to configure — the port is found at startup.

To check which port it landed on:

```bash
python -m hardware_test.find_ap
```

```
  /dev/cu.usbmodem101 ... AP  frame_ok=19 bad_ck=0 over 3s
```

### 2. Check the wheels turn

Put a car on the floor with room around it, and give it the id you flashed
into its firmware:

```bash
python -m hardware_test.check_motion --cars 3
```

It drives forward, backward, counter-clockwise, then clockwise, pausing
between each so you can watch.

### 3. Connect the camera

Plug the camera in and point it at the area you want to drive in. On a Mac,
Photo Booth is the quickest way to see what it sees.

There is no strict requirement on angle or height. The workspace is derived
from the view itself, so all that matters is that the camera covers the
area you want to work in.

### 4. Run the demo

```bash
python examples/navigation.py --config config/car_v4.yaml
```

On startup it prints the workspace it measured and the cars it found:

```
[main] Workspace ready: 90.2 × 51.4 cm
[main] Detected cars: [3]
```

### 5. Drive it

In the window: draw a curve on the canvas and the robot follows it, or
click a point and it plans a path there. Click a car to select it. The
sidebar has a speed slider and a "rotate to" box. `C` clears the current
task, `Space` stops, `Escape` quits.

There is also an HTTP API on port 8080 for driving it from your own code:

```bash
curl -X POST http://localhost:8080/goto \
     -H "Content-Type: application/json" \
     -d '{"x": 30, "y": 20, "theta": 45}'
```

Full API in [docs/navigation_api.md](docs/navigation_api.md).

---

## Concepts

MicroMVP is built from three components.

**Controller** does the low-level motion control of a single robot: follow
this path, rotate to that heading. One controller per robot.

**Environment** is the world the robots act in. It changes as they move,
and it gives you observations — where each robot is and which way it faces.
`SimEnv` simulates it; `RealEnv` is the real one, seen through a camera.

**Coordinator** is in charge overall. It takes the observations from the
environment and hands each controller what it needs, and it handles
everything that involves more than one robot — collision avoidance, path
planning, deciding who goes where.

### Environment — takes actions, returns observations

It applies the actions it is given, which changes the state of the world,
and it reports observations of that state.

```python
observations = env.observe()               # {car_id: RobotObservation(x, y, theta, t)}
env.apply_actions({3: Action(0.5, 0.5)})   # left/right wheel thrust, -1..1
```

That is the entire interface. How the state changes and where the
observations come from is up to the implementation: `SimEnv` steps a model
in memory, `RealEnv` reads poses from the overhead camera and sends wheel
commands to the AP. Anything implementing `observe` / `apply_actions`
works.

`RealEnv` also derives the workspace itself — there are no calibration
markers on the floor. It fits the ground plane from the markers it can see,
projects the camera's field of view onto that plane, and takes the largest
rectangle inside it. Details in
[src/micromvp/env/real_env/README.md](src/micromvp/env/real_env/README.md).

### Controller — low-level motion control for one robot

One instance per robot. You give it a task — a path to follow, a heading to
rotate to — and it keeps its own state while carrying that task out, one
`step()` per observation.

```python
controller = NavigationController.from_config(robot_id, ws_config, cfg)
controller.set_path([(10, 10), (20, 15), (30, 10)])
action = controller.step(observation)
```

Shipped controllers:

| Controller | What it does |
|---|---|
| `NavigationController` | Pure pursuit + cross-track-error PD, with in-place rotation. The one the example uses. |
| `PurePursuitFollowPathController` | Plain pure pursuit. |
| `PurePursuit_PD_FollowPathController` | Pure pursuit with a PD correction term. |
| `StanleyFollowPathController` | Stanley steering. |
| `TargetFollowController` | Chase a moving point. |
| `WASDController` | Keyboard driving. |

To write your own, subclass `Controller` and implement three methods:

```python
class MyController(Controller):
    def update(self, observation):      # absorb the new pose
        self._car_state.x = observation.x
        self._car_state.y = observation.y
        self._car_state.theta = observation.theta

    def calculate_action(self):         # decide what to do about it
        return Action(left_speed=0.0, right_speed=0.0)

    def step(self, observation):
        self.update(observation)
        return self.calculate_action()
```

### Coordinator — coordination across robots

Processes the observations from the environment, hands each controller what
it needs, and collects the actions back. Anything spanning more than one
robot belongs here: collision avoidance, path planning, formations, task
assignment. It is also what the GUI talks to.

```python
actions = coordinator.process(observations)
env.apply_actions(actions)

car_states = coordinator.gather_car_state()          # for rendering
drawings = coordinator.get_additional_drawings()     # overlays
```

Shipped coordinators: `NavigationCoordinator` (single robot, obstacle
avoidance, HTTP API), `KeyboardCoordinator`, `FollowPathCoordinator`,
`FormationCoordinator`.

### Putting it together

The whole main loop, with nothing left out:

```python
from micromvp.config import load_config
from micromvp.controller import NavigationController
from micromvp.coordinator import NavigationCoordinator
from micromvp.env import RealEnv

cfg = load_config("config/car_v4.yaml")

env = RealEnv(cfg)
env.start(wait_for_ready=True, timeout=10.0)
ws_config = env.workspace_config

controllers = {
    rid: NavigationController.from_config(rid, ws_config, cfg)
    for rid in ws_config.car_id_list
}
coordinator = NavigationCoordinator.from_config(ws_config, controllers, cfg)

while True:
    observations = env.observe()
    actions = coordinator.process(observations)
    env.apply_actions(actions)
    time.sleep(1 / ws_config.frequency)
```

`examples/navigation.py` is this plus a GUI and a render timer.

### Which layer do I change?

| I want to… | Change |
|---|---|
| use different hardware or a different tracker | Environment |
| change how a robot follows a path | Controller |
| coordinate several robots, assign tasks | Coordinator |
| change the planner | `planner.name` in the config |
| change a physical dimension or a gain | the config, never code |

---

## Configuration

One YAML file describes one physical setup, completely. To reproduce a
deployment, copy the file.

```bash
cp config/car_v4.yaml config/my_table.yaml
python examples/navigation.py --config config/my_table.yaml
```

Every module reads its fields out of that file, so there are no per-module
config classes to learn. `config/car_v4.yaml` is commented throughout and
is the reference.

| Section | Covers |
|---|---|
| `car` | ArUco dictionary, marker edge length and height, body size, axle offset, wheel base |
| `obstacle` | ArUco dictionary, marker size and height, and each marker id's polygon |
| `camera` | device, resolution, fps, calibration file, preview |
| `workspace` | margins and how steady the estimate must be before it locks |
| `tracking` | per-car outlier rejection |
| `actuation` | serial port, baud, send rate, wheel inversion |
| `runtime` | main loop frequency |
| `control` | speed ceiling, lookahead, goal tolerance, rotation behaviour |
| `navigation` | web API port, active robot, planner safety margin |
| `planner` | which planner, and its parameters |

Fields are **required**, not optional. A missing one stops startup and says
exactly what to do:

```
missing required field 'car.marker_size_mm'
  needed by : ArucoObserver
  config    : config/my_table.yaml
  add it to the 'car' section.
  that section currently has: aruco_dict, axle_offset_cm, body_height_cm, ...
```

A field the file contains but nothing reads — usually a typo — is reported
on startup too.

### Measuring the physical fields

- **`car.marker_size_mm`** — the side of the marker's *black border*, not
  including the white quiet zone. ArUco infers distance from apparent size,
  so this scales every distance the system reports.
- **`car.marker_height_cm`** — how far the marker plane sits above the
  floor, i.e. the height of the car.
- **`obstacle.marker_height_cm`** — same for obstacle blocks. Use `0.0` if
  the markers lie flat on the ground.

Cars and obstacles must use different ArUco dictionaries (4x4 and 5x5 by
default), so that a car and an obstacle sharing an id stay distinguishable.

---

## Path planning

`NavigationCoordinator` plans around obstacles when a planner is available.

- **`planner.name: straight`** — straight line from start to goal. Ignores
  obstacles. Always available.
- **`planner.name: rvg`** — a rotational visibility graph, which plans in
  SE(2) and so accounts for the robot's shape *and* heading. Needs the RVG
  extension built and importable.

If `rvg` is selected but not importable, the system warns once at startup
and falls back to straight lines.

### Installing RVG

RVG is not vendored here. It is a separate C++ project
([arc-l/rvg](https://github.com/arc-l/rvg)) with Python bindings, and
building it is a fair amount of work — budget half an hour, and note that
its own README describes the macOS bindings as still under development.

```bash
git clone --recurse-submodules https://github.com/arc-l/rvg.git
cd rvg

# system libraries (Ubuntu; use the brew equivalents on macOS)
sudo apt install libboost-all-dev libgmp-dev libmpfr-dev \
                 libtinyxml2-dev libeigen3-dev

conda activate micromvp     # build against the env you run MicroMVP in
pip install -e .
```

Check it took:

```bash
python -c "from rvg import vertex, polygon, rvg; print('ok')"
```

Import it from a directory that is not the RVG checkout. A bare `rvg/`
folder on the path imports as an empty namespace package, so plain
`import rvg` can succeed while the real extension is missing.

With it installed, the path-planning tests run:

```bash
pytest -m rvg
```

### DynamicRVG simulation

`examples/dynamic_rvg_simulation.py` runs a persistent DynamicRVG session
against `SimEnv` and the normal `NavigationController`. It updates the planner
from simulated pose observations, executes each temporary segment through the
differential-drive model, and lets the controller decide when the final goal
is reached.

```bash
python examples/dynamic_rvg_simulation.py
```

Add `--gui` for a live MicroMVP canvas:

```bash
python examples/dynamic_rvg_simulation.py --gui
```

The simulation exposes the same planner controls as
`mainDebugDynamicRVG`, for example:

```bash
python examples/dynamic_rvg_simulation.py \
  --mode buffered_delta_graph_merge \
  --strategy quadtree_frontier_astar_information \
  --scan-mode center --resolution 36 --num-threads 1
```

Use `--help` for the weight, iteration-limit, quadtree, and information-gain
options.

The GUI shows the simulated scan boundary, obstacle geometry, current and
previous DRVG segments, controller target, measured trajectory, temporary
goal, and final goal heading. Click the canvas to plan to a new goal from the
current robot pose. Press `R` to restart, `Space` to pause, or `Escape` to
close the simulation.

If the RVG extension was built in a checkout rather than installed, add its
build directory to `PYTHONPATH`:

```bash
PYTHONPATH=/path/to/drvg/code/build-python python examples/dynamic_rvg_simulation.py
```

The run writes `trajectory.png` and one planner graph per planning step under
`simulation_output/dynamic_rvg/`.

### DynamicRVG on real hardware

`examples/dynamic_rvg_navigation.py` uses the same persistent planner with
`RealEnv`. At startup it captures one fixed obstacle snapshot from the ArUco
observer. During a run, every measured robot pose is passed to DynamicRVG, one
temporary segment is executed at a time. Its SE(2) configurations are executed
as explicit in-place turns and individual straight segments, including the
terminal heading. Each translation can drive forward or backward: the executor
chooses the direction requiring less turning while respecting the endpoint
headings. A short backward step therefore does not force two half-turns.
Reverse following keeps the measured body heading in the physical frame and
swaps/negates virtual forward wheel commands.

As in `examples/navigation.py`, a turn immediately preceding a drive hands off
once it enters heading tolerance (`ROTATION_STABLE`). Final or standalone turns
still wait for the configured settled completion. Intermediate turns are not
converted into offset XY waypoints, and pure pursuit never looks ahead across
a planned corner. The status line and console report turn error and drive
direction. Rotation stopping uses the latest measured heading to avoid EMA
lag during a turn; the path follower's smoothed heading is synchronized at
handoff.

The hardware demo uses `navigation.execution_collision_check: false`: it
executes the turn/drive sequence without a second collision-based rejection.
DRVG planning continues to use the inflated robot and padded obstacles.
Explicit in-place turns, wrapped heading control, and measured-pose replanning
remain in use. Padding supplies extra planning clearance; it does not establish
that every additional differential-drive turn fits.

To enable the optional executor checks, set
`navigation.execution_collision_check: true`. These validate inflated swept
footprints before execution and at measured poses, including an arc-deviation
bound between sampled rotation angles.

The GUI/recording obstacle outlines and visible-area shading both use original
detected obstacles. The display performs an independent scan without building
another visibility graph. Planner visibility and path selection still use
`navigation.obstacle_padding_cm`, so the display can show a physical strip of
free space that the planner excludes as clearance.

The deployment uses `navigation.robot_footprint: circle`. Its centre is the
wheel axle, explicitly passed as RVG's robot centre at local `(0, 0)`.
The radius is the maximum axle-to-corner distance of the configured physical
body, multiplied by `robot_geometry_scale`. With the current dimensions this
is 6.46 cm (12.91 cm diameter). RVG receives a 72-sided polygon circumscribed
around that disk, so polygon approximation does not cut into the turning
envelope. Use `robot_footprint: polygon` to restore the rectangular proxy.

The circle is only a collision envelope. Measured poses and graph states retain
a full 0–360 degree heading range; dynamic RVG disables geometry-symmetry
folding, and 0 and 180 degrees remain distinct. Reverse driving retains the
physical heading. In circle mode the executor omits unnecessary intermediate
heading alignments but honors the segment endpoint heading.

The current rotation weight is 5.0, with translation weight 1.0 and angles in
radians: a 90-degree search rotation costs about as much as 7.85 cm of travel.
Angular distance wraps at 360 degrees (359→1 costs 2 degrees). Search weighting
does not account for every additional turn needed by the physical executor.
The circular envelope trades narrow-passage access for space to turn.

The deployment enables `planner.rvg.optimal: true`. This selects DRVG's optimal
layer-connection construction, not a guarantee of globally optimal navigation
through an unknown world. The dynamic binding must expose
`setOptimalRvgConstruction`; after updating the DRVG source, rebuild with:

```bash
cmake --build ~/drvg/code/build-python --target rvg -j32
```

Run it from the MicroMVP checkout with the DRVG extension installed or on
`PYTHONPATH`:

```bash
PYTHONPATH=/path/to/drvg/code/build-python python examples/dynamic_rvg_navigation.py \
  --config config/car_v4.yaml
```

To use one perspective-aligned GUI with scan coverage and planner drawings
over the live camera image, run the alternate overlay entry point with the
same arguments:

```bash
PYTHONPATH=/path/to/drvg/code/build-python python examples/dynamic_rvg_navigation_overlay.py \
  --config config/car_v4.yaml
```


For a rectangular top-down presentation, use the overlay's bird's-eye view:

```bash
PYTHONPATH=/path/to/drvg/code/build-python python examples/dynamic_rvg_navigation_overlay.py \
  --config config/car_v4.yaml --view birdseye --record recordings/demo-birdseye.mp4
```

This warps the camera's floor plane into the locked workspace rectangle and
preserves its physical width-to-height ratio. Goal clicks and planner overlays
use the same transformation. The recording captures the rectified view.
`--view perspective` (the default) shows the original camera image. Raised
objects retain parallax; rectification does not change camera calibration.

For a physically top-down view, mount the camera above the workspace with the
lens pointing vertically down, and level its mounting platform in both
directions. A known rectangle on the floor should have parallel opposite edges
and similar apparent lengths for opposite sides in the original perspective
view. After moving the camera, restart the demo to estimate a new workspace.
Recalibrate if the camera or lens focus changes.

To record only the camera view and planner overlays (without the sidebar,
desktop cursor, or audio), add `--record` to the overlay command:

```bash
PYTHONPATH=/path/to/drvg/code/build-python python examples/dynamic_rvg_navigation_overlay.py \
  --config config/car_v4.yaml --record recordings/demo.mp4
```

Recording is armed at startup and starts only when a goal click is accepted.
It includes the obstacle rescan, planning and movement, then automatically
finalizes when the goal is reached (including its final heading). Idle time
before or after a run is excluded. Stop/cancel, planner failure, manual obstacle
recapture, or closing the window also finalizes the current clip. A temporary
tracking pause stays in the same clip.

With `--record recordings/demo.mp4`, the first goal uses `demo.mp4`; later
goals use `demo_002.mp4`, `demo_003.mp4`, and so on. Selecting a new goal
during a run closes the old clip and starts a new one. Existing numbered files
are skipped, never overwritten; choose an unused base filename at launch.
Output uses the displayed camera/rectified resolution at 30 fps independently
of window size. Encoding runs on a separate thread, and stale frames repeat
if the GUI stalls. `recordings/` is ignored by Git.

The real-hardware command accepts the same `--mode`, `--strategy`,
`--scan-mode`, resolution, weight, iteration-limit, quadtree, and
information-gain options as the simulation command. Its default
`graph_merge` mode retains the accumulated graph across measured-pose steps.

Each initialized goal run also writes `measured_run_*.csv` and a matching
metadata JSON in the output directory, automatically. The CSV includes
timestamps, measured axle x/y, wrapped and unwrapped heading, wheel commands,
segment/motion targets, and controller state. It is flushed every second and
closed at completion, cancellation, failure, or exit. Continuous heading
restarts after tracking loss or an observation gap over 0.5 seconds; use the
`tracking_epoch` column to distinguish intervals. Commands are requested
thrust, not motor feedback. The metadata links the prepared route when present.

### Preplan the full route, then execute it

For the fixed-table demo, prepare the entire DRVG route before moving:

```bash
PYTHONPATH=/path/to/drvg/code/build-python python examples/dynamic_rvg_navigation_overlay.py \
  --config config/car_v4.yaml --view birdseye --execution-mode preplanned \
  --record recordings/demo-preplanned.mp4
```

A goal click captures obstacles and starts a background virtual DRVG run,
advancing between exact planned endpoint poses until a final-goal path is found.
The car stays stopped while the complete route is prepared. The camera and
Stop button remain responsive. Preparation must finish successfully before
any trajectory is executed; `--preplan-timeout 30` limits preparation time.

The saved segments then run with live camera pose feedback and the existing
turn/forward/reverse controller. There are **no measured-pose DRVG replans**
during playback. At each segment boundary the car stops, the original-obstacle
visibility display updates from its measured position, and it pauses for
`--scan-pause 0.5` seconds before continuing. The GUI labels this as
**preplanned playback**. A JSON copy of each prepared route is saved in the
output directory.

Keep the obstacle layout and starting car position fixed during preparation
and execution. This mode separates planning from tracking; it does not change
the car's kinematic limits. Stop/cancel, a new goal, or obstacle recapture
discards the prepared route. Recordings still start at the accepted goal click
and finish at goal completion or cancellation.

This setup uses a fixed overhead camera, a locked workspace, an obstacle
snapshot, and live car localization; it is not live SLAM. DRVG's scans are
software visibility queries against that snapshot. Use
`--execution-mode live` for the previous measured-pose replanning behavior.

Workspace startup first averages each marker's image corners over
`workspace.corner_smoothing_frames` (30 by default), then solves marker poses
from those averaged corners. Any corner more than
`workspace.corner_motion_tolerance_px` (2 pixels) from its window mean resets
accumulation; a changed marker set also starts a new window. These estimates
must then pass the existing 30-frame dimension, origin and tilt stability checks.
This filtering applies only before workspace lock; live robot tracking continues
to use current-frame marker observations.

Startup first shows a camera setup window with detected car IDs, the current
workspace dimensions, and the sample count. Filling the sample window (for
example, 30/30) is not enough: the estimates must also pass the stability limits.
If startup times out, the preview stays open. Keep the camera and markers still,
check focus and calibration, then click **Retry workspace lock**. **Cancel**
closes the environment. The navigation window opens only after the workspace
locks and a car is detected. Use `--timeout 30` for a longer initial wait.

The robot remains stopped until you click a goal on the GUI canvas. Enter a
goal heading before clicking if needed. `Space` or `C` immediately stops the
robot and cancels the run; `O` stops the robot and replaces the fixed obstacle
snapshot. Camera tracking loss and any planner failure also produce a stop
command. Diagnostic graph PNGs are disabled by default for real hardware.
Use `--planner-drawings` to write them beneath
`navigation_output/dynamic_rvg/`. This invokes synchronous Matplotlib rendering
after each plan and can add seconds before the robot moves. Live camera
overlays and video recording work with PNG output disabled.
The console reports `planning_ms` (planner plus path adaptation) and, when
enabled, `diagnostic_plot_ms` separately. Each new goal also captures obstacles
for `--obstacle-capture-seconds` (default 1 second); segment completion includes
the configured heading-alignment tolerance and settling time.

---

## Solver geometry inspector

Capture a camera-only snapshot and view RVG's native geometry:

```bash
PYTHONPATH=/path/to/drvg/code/build-python python examples/inspect_drvg.py \
  --config config/car_v4.yaml --show
```

It exports `solver.png`, `solver.svg`, `geometry.json`, and `scene.json`
under a timestamped `diagnostics/drvg/` directory. The four panels show the
solver environment, the physical/scaled/angular bounding footprints about the
axle, native legal axle positions, and a close-up at the captured pose.
The bounds, grown obstacles and shrunken border come from RVG's native
`Layer` API, and the green/red map samples `Layer.legalConfig()`.
This is a frozen snapshot, not a live control window, and no serial/motor
connection is opened.

Replay it at another heading without opening the camera:

```bash
PYTHONPATH=/path/to/drvg/code/build-python python examples/inspect_drvg.py \
  --scene diagnostics/drvg/<timestamp>/scene.json --heading 180 --show
```

The current `workspace.margin_cm: 0.0` uses the full inscribed camera view.
RVG still shrinks that boundary by its robot bounding footprint when determining
legal axle positions. Adding a workspace margin would apply another inset.

---

## Camera alignment helper

Run this before the navigation demo to physically aim the camera straight down:

```bash
python examples/align_camera.py --config config/car_v4.yaml
```

The helper opens only the camera; it does not connect to the AP or motors and
does not need the DRVG extension. Keep flat car/obstacle markers in view.
The arrow and text tell you which way to **tilt the lens**, using directions
in the live image. Make a small adjustment, then pause for the marker corner
averaging window to settle. Height and centering are manual; rolling the camera
around its optical axis does not make it more perpendicular to the floor.

A green screen and **STOP — ALIGNED** appear after 1.5 continuous seconds with:

- floor-normal tilt at most 3 degrees;
- every projected workspace corner within 3 degrees of a right angle;
- opposite projected side lengths differing by at most 5 percent.

The helper continues measuring after green. Camera motion, lost/stale frames,
or an out-of-tolerance estimate clears the green state. These are estimates
from the configured camera calibration, marker sizes and heights; the tool
does not recalibrate the camera. Secure the mount when green, press **Esc**,
and then start the demo so it locks a fresh workspace.

Optional thresholds: `--tilt-tolerance 3 --corner-tolerance 3
--edge-tolerance 5 --hold-seconds 1.5`. Angles are degrees and edge tolerance
is a percentage. Use `--help` for details.

---

## Camera calibration

The vision system needs your camera's intrinsics. `config/camera.yaml`
ships with a working set, but it describes *our* camera — recalibrate for
yours.

```bash
python calibration/generate_board.py     # produces a ChArUco board to print
python calibration/calibrate_camera.py   # capture views, solve, write camera.yaml
```

See [calibration/README.md](calibration/README.md).

Recalibrate when you use a new camera.

---

## Troubleshooting

### A car does not move

Check the battery first — a car with a flat battery still shows up in the
camera and still receives commands, it just cannot drive. Plug in the
charging cable and run the motion check again:

```bash
python -m hardware_test.check_motion --cars 3
```

If it still does not move, the `CAR_ID` in `xiao/xiao_1_8_ESP_NOW.ino` does
not match the id you are testing. The AP broadcasts to every car at once and
each one picks out its own slot by id, so a mismatch looks exactly like a
dead car. Note that ESP-NOW broadcasts are not acknowledged: the AP
reporting `send_ok` means it sent the packet, not that any car heard it.

If it moves the wrong way, set `actuation.invert_left_wheel` /
`invert_right_wheel`.

### The workspace never locks

The lock counter climbs, drops to zero, and repeats. Startup times out with
`workspace not ready`.

Look at the camera in Photo Booth, or any other preview, for a few seconds.
If the picture keeps going soft and then sharp again, the camera is hunting
for focus — plain carpet or a bare table gives autofocus nothing to lock
onto. Blurred frames detect no markers, and one frame without markers
clears the accumulated window, so the count never reaches
`workspace.lock_frames`. On cameras that expose standard focus controls, set
`camera.autofocus: false` and choose `camera.focus_absolute` in the deployment
YAML. The observer reapplies both settings whenever it opens the camera. Put a
sheet of white paper under the workspace if the camera still cannot focus
reliably.

If the picture is steady and sharp, the cause is something else.

---

## Repository layout

```
config/            one YAML per deployment, plus camera intrinsics
examples/          navigation.py — the worked example
hardware_test/     bring-up checks you run by hand against real hardware
calibration/       ChArUco board generation and camera calibration
src/micromvp/
  config.py        the config loader every module reads through
  core/            data models, differential-drive kinematics, path patterns
  env/             Environment interface, SimEnv, RealEnv
  controller/      per-robot control algorithms
  coordinator/     multi-robot orchestration and the GUI bridge
  gui/             PyQt6 window, canvas, sidebar
xiao/              ESP32 firmware — car and AP
tests/             pytest suite, no hardware needed
docs/              web API, GUI spec, original design notes
```

---

## Coordinate system

- Origin at the bottom-left of the workspace, `(0, 0)`
- **X** right, **Y** up
- **Theta** 0° along +X, increasing counter-clockwise
- Distances in centimetres; wheel commands normalised to `-1..1`

The workspace origin is tied to the camera's field of view, not to anything
on the floor. That is why the estimate is locked once and then frozen — move
the camera and every coordinate from before the move becomes meaningless.

---

## Tests

```bash
pytest              # 63 tests, no hardware required
pytest -m rvg       # additionally requires the RVG plugin
```
