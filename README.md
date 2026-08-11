# MuJoCo Maze Generator + Full-Coverage Exploring Bot

Two independent pieces, designed to be used together or standalone:

| File | Purpose |
|---|---|
| `maze_generator.py` | Builds a random maze and exports it as a MuJoCo model (`.xml`) you can open in the MuJoCo viewer. |
| `maze_explorer.py` | Drops a robot **anywhere** in that maze and drives it to visit every cell, using only its own onboard sensors, then saves the map it built. |
| `maze_localize_and_navigate.py` | Drops a robot **anywhere**, figures out *where it is* using only its sensors and a known map, then drives the shortest path to a destination the user picks. |
| `maze_viewer.py` | Shared helper: opens a live MuJoCo viewer window and paces physics stepping to real time, so you can watch the robot move during both exploring and navigating. |
| `run_demo.py` | Runs the generate + explore steps back to back. |

## Watching the robot move live

By default, both `maze_explorer.py` and `maze_localize_and_navigate.py` now
open a live MuJoCo viewer window and drive it in real time — you'll see the
robot physically slide through the maze while it explores, localizes, and
navigates, instead of the simulation running silently in the background and
only producing a PNG/JSON at the end.

Flags on both scripts:
- `--no-render` — run fully headless (no window), like before.
- `--no-realtime` — step as fast as possible instead of pacing to real time
  (the window still updates, just faster than watchable).
- `--hold-open N` — keep the window open for `N` seconds after finishing so
  you can see the final state (default `3.0`).

Notes:
- You need an actual display for this (a local machine, or a remote session
  with X11 forwarding / VNC). If none is detected, both scripts print a
  warning and automatically fall back to headless — they won't crash.
- On macOS, the live viewer must be launched with `mjpython` instead of
  `python3` (a MuJoCo/GLFW requirement), e.g.
  `mjpython maze_explorer.py --xml maze.xml --rows 8 --cols 8`.

## 1. Generate a maze

```bash
python3 maze_generator.py --rows 10 --cols 10 --seed 1 --out maze.xml
```

This writes `maze.xml` (the MuJoCo model) and `maze_ground_truth.json` (the true
wall layout — useful for debugging/validation, but the robot never reads it).

**View it interactively** (needs a display):
```bash
python3 -m mujoco.viewer --mjcf=maze.xml
```
or in Python:
```python
import mujoco, mujoco.viewer
m = mujoco.MjModel.from_xml_path("maze.xml")
d = mujoco.MjData(m)
mujoco.viewer.launch(m, d)
```

Maze generation uses randomized recursive backtracking, which always produces
a **perfect maze**: every cell is reachable, and there are zero loops (it's a
spanning tree over the grid). That property is what makes complete,
guaranteed traversal possible with a simple, memoryless control rule (see below).

The generated model includes:
- Grid-aligned walls (boxes) matching the maze layout, plus an outer boundary.
- A cylindrical robot body that slides freely in X/Y (two velocity-actuated
  slide joints — holonomic, no heading to worry about).
- Four rangefinder sensors on the robot (`rf_north/south/east/west`), each
  aimed along a fixed world axis, that report the distance to the nearest
  wall in that direction (or the sensor cutoff if the way is clear).

## 2. Explore + map the maze

```bash
python3 maze_explorer.py --xml maze.xml --rows 10 --cols 10 \
    --start-row 4 --start-col 7 --out discovered_map.json --png discovered_map.png
```

`--start-row/--start-col` place the robot at **any** cell — the traversal
algorithm doesn't care where it starts.

### How it explores
The robot uses the classic **right-hand wall-following rule**: at each cell,
try to turn right first, then straight, then left, then U-turn — always
taking the first direction its live rangefinders report as open. It never
looks at the ground-truth maze file.

Because the maze is guaranteed to be a simple tree (no loops), this rule is
mathematically guaranteed to visit every reachable cell and return to the
start — regardless of starting cell. The script stops automatically once
that happens (with a generous move-count safety cap as a fallback).

### What gets saved
- `discovered_map.json` — every cell the robot visited, the walls it sensed
  around each one (N/E/S/W open or blocked), and the full path taken.
- `discovered_map.png` — a top-down plot of the reconstructed maze walls
  with the robot's path traced over it.

## 3. Localize + navigate to a destination

Once you have a map (from step 2, or the ground-truth file), you can drop the
robot on any cell — without telling the algorithm where that is — and have
it figure out its own location, then drive to wherever you want:

```bash
python3 maze_localize_and_navigate.py --xml maze.xml --map discovered_map.json \
    --spawn-row 5 --spawn-col 2 --dest-row 0 --dest-col 7
```

Omit `--spawn-row/--spawn-col` to drop it on a random cell; omit
`--dest-row/--dest-col` to be prompted for a destination interactively.

### How it localizes ("kidnapped robot" problem)
The robot's control code only ever calls two methods: `sense()` (read the 4
rangefinders) and `step(direction)` (attempt to move one cell — refuses if
its own sensor says that way is blocked). It never reads its true simulator
position.

1. Take one sensor reading and find every map cell whose wall pattern
   matches it. Often (especially in a straight corridor) several cells share
   the same local pattern, so this is ambiguous.
2. If ambiguous, pick a direction that's actually open right now and move
   one cell, using a greedy minimax rule that favors whichever open
   direction will split the remaining candidates the most (similar to an
   optimal "20 questions" strategy). Take a new reading and keep only the
   candidates whose map neighbor in that direction matches it.
3. Repeat until exactly one candidate remains — that's the robot's real
   location, guaranteed (since its true cell survives every filtering step
   by construction).

### How it plans + navigates
With a known current cell and a user-chosen destination, it runs **A\*** over
the map graph (uniform edge cost, Manhattan-distance heuristic — equivalent
to BFS on this unweighted grid, shown here as A* since it generalizes to
weighted/irregular graphs too) to get the shortest cell-to-cell path, then
drives it step by step, double-checking with its sensor before every move
that the map's claimed passage is really open.

`localize_and_navigate.png` visualizes the run: which cells were still
candidates at each stage of localization (blue), the planned path (green
dashed), and the path actually driven (red).

## 4. Run generation + exploration in one go

```bash
python3 run_demo.py --rows 10 --cols 10 --seed 3 --random-start
```

## Notes / things you can tune
- `--cell-size` in `maze_generator.py` controls the physical scale.
- Sensor range, robot radius, wall thickness, and controller gains
  (`kp`, `vmax`, `damping`) are set to sensible defaults for `cell-size=1.0`;
  shrink them proportionally if you use a much smaller/larger cell size.
- For an unknown/irregular arena (not a generated perfect maze), swap the
  wall-follower in `MazeExplorer.explore()` for a frontier-based exploration
  policy — wall-following only guarantees full coverage on simply-connected
  (loop-free) mazes like the ones `maze_generator.py` produces.
