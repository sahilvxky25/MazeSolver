"""
maze_localize_and_navigate.py
------------------------------
Third stage of the pipeline (after maze_generator.py builds the maze and
maze_explorer.py maps it):

  1. The robot is dropped at an ARBITRARY, UNKNOWN cell in a maze it already
     has a map of (e.g. the discovered_map.json produced by
     maze_explorer.py, or the ground-truth file — anything with the same
     shape).
  2. It LOCALIZES itself: using only its own onboard rangefinder readings
     (never its simulator ground-truth position), it works out which cell
     of the known map it is standing in. This is the classic "kidnapped
     robot problem" — solved here by matching the local wall pattern
     against the map, and, if that's ambiguous, taking a few real moves and
     using the resulting reading sequence to narrow the candidates down to
     one.
  3. Once localized, the user picks a destination cell. The robot plans the
     shortest path there with A* over the map graph, then physically drives
     that path.

Usage:
    python3 maze_localize_and_navigate.py --xml maze.xml --map discovered_map.json \\
        --spawn-row 5 --spawn-col 2 --dest-row 0 --dest-col 7

    # omit --dest-row/--dest-col to be prompted for a destination interactively
    # omit --spawn-row/--spawn-col to be dropped on a random cell
"""

import argparse
import heapq
import json
import random
import time

import numpy as np
import mujoco

from maze_viewer import LiveViewer

N, S, E, W = "N", "S", "E", "W"
OPPOSITE = {N: S, S: N, E: W, W: E}
DIR_OF_DELTA = {(-1, 0): N, (1, 0): S, (0, 1): E, (0, -1): W}
SENSOR_NAME = {N: "rf_north", S: "rf_south", E: "rf_east", W: "rf_west"}


# ============================================================================
# 1. Map loading — accepts either maze_explorer.py's discovered_map.json
#    ({"walls": {"r,c": {"N": bool, ...}}}) or maze_generator.py's
#    ground-truth json ({"grid": {"r,c": ["N","E",...]}}). Both normalize to
#    the same graph representation: {(r, c): set_of_open_directions}.
# ============================================================================

def load_map_graph(path):
    with open(path) as f:
        data = json.load(f)
    rows, cols = data["rows"], data["cols"]
    graph = {}
    if "walls" in data:
        for key, dirs in data["walls"].items():
            r, c = map(int, key.split(","))
            graph[(r, c)] = {d for d, open_ in dirs.items() if open_}
    elif "grid" in data:
        for key, dirs in data["grid"].items():
            r, c = map(int, key.split(","))
            graph[(r, c)] = set(dirs)
    else:
        raise ValueError(f"Unrecognized map file format: {path}")
    return graph, rows, cols


def signature(graph, cell):
    """The set of open directions at `cell` — what a sensor reading looks like."""
    return frozenset(graph.get(cell, set()))


def neighbor(cell, direction):
    r, c = cell
    dr, dc = {N: (-1, 0), S: (1, 0), E: (0, 1), W: (0, -1)}[direction]
    return (r + dr, c + dc)


# ============================================================================
# 2. Physical/sim robot interface. Deliberately narrow: sense() and step()
#    are the ONLY things the localization/planning logic is allowed to use.
#    Nothing here ever hands back "you are at cell (r,c)" — that's exactly
#    the thing the localizer has to work out for itself, the same as it
#    would on a real robot with no GPS.
# ============================================================================

class MazeRobotSim:
    def __init__(self, xml_path, cell_size=1.0, spawn_row=0, spawn_col=0,
                 open_threshold_frac=0.6, render=True, realtime=True):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.cell_size = cell_size
        self.body_base = self.model.body("robot").pos[:2].copy()

        cutoff = self.model.sensor_cutoff[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "rf_east")
        ]
        self.open_threshold = cutoff * open_threshold_frac

        # Live MuJoCo viewer — makes both localization moves and the final
        # navigation drive visible in real time instead of running silently
        # in the background.
        self.live = LiveViewer(self.model, self.data, enabled=render, realtime=realtime)

        self._true_cell = (spawn_row, spawn_col)  # only for verification/plots
        self._set_world_pos(*self._cell_center(spawn_row, spawn_col))
        self.trajectory_xy = [self.world_pos().copy()]
        self.live.sync(self.model, self.data)  # show the spawn pose right away

    # -- internal geometry (private: real robot code wouldn't need this at
    #    all since it would just command wheel velocities, but the sim
    #    needs it to know pixel coords for a given logical cell) --
    def _cell_center(self, r, c):
        x = c * self.cell_size + self.cell_size / 2.0
        y = -(r * self.cell_size + self.cell_size / 2.0)
        return x, y

    def _set_world_pos(self, x, y):
        self.data.qpos[:2] = np.array([x, y]) - self.body_base
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

    def world_pos(self):
        return self.data.qpos[:2] + self.body_base

    # -- the only two methods the localizer/planner may call --
    def sense(self):
        """Live rangefinder reading -> {'N': open?, 'S': ..., 'E': ..., 'W': ...}"""
        reading = {}
        for d, sname in SENSOR_NAME.items():
            val = self.data.sensor(sname).data[0]
            reading[d] = bool(val > self.open_threshold)
        return reading

    def step(self, direction, kp=6.0, vmax=2.5, tol=0.02, max_steps=4000):
        """
        Attempts to physically move one cell in `direction`. Returns False
        immediately (no movement) if the robot's own sensor says that way
        is blocked — a real robot can't drive through a wall just because
        a map says the passage should be there.
        """
        if not self.sense()[direction]:
            return False

        r, c = self._true_cell
        dr, dc = {N: (-1, 0), S: (1, 0), E: (0, 1), W: (0, -1)}[direction]
        target_cell = (r + dr, c + dc)
        target_xy = np.array(self._cell_center(*target_cell))

        act_x = self.model.actuator("act_x").id
        act_y = self.model.actuator("act_y").id
        for _ in range(max_steps):
            pos = self.world_pos()
            err = target_xy - pos
            if np.linalg.norm(err) < tol:
                break
            vel = np.clip(kp * err, -vmax, vmax)
            self.data.ctrl[act_x] = vel[0]
            self.data.ctrl[act_y] = vel[1]
            mujoco.mj_step(self.model, self.data)
            self.live.sync(self.model, self.data)  # render this physics step
            self.trajectory_xy.append(pos.copy())
        self.data.ctrl[act_x] = 0.0
        self.data.ctrl[act_y] = 0.0
        self.data.qvel[:2] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.live.sync(self.model, self.data)

        self._true_cell = target_cell  # sim bookkeeping only
        self.trajectory_xy.append(self.world_pos().copy())
        return True

    def debug_true_cell(self):
        """For verification/plotting ONLY. The localizer never calls this."""
        return self._true_cell

    def close(self, hold_open=3.0):
        """Close the live viewer window (optionally holding the last frame)."""
        self.live.close(hold_open=hold_open)


# ============================================================================
# 3. Localization — the kidnapped robot problem.
# ============================================================================

class LocalizationError(Exception):
    pass


class Localizer:
    def __init__(self, graph):
        self.graph = graph
        self._sig_index = {}
        for cell, dirs in graph.items():
            self._sig_index.setdefault(frozenset(dirs), []).append(cell)

    def _candidates_matching(self, reading):
        sig = frozenset(d for d, open_ in reading.items() if open_)
        return set(self._sig_index.get(sig, []))

    def _choose_disambiguating_direction(self, candidates, open_dirs):
        """
        Greedy minimax choice among directions the robot can actually move
        right now: pick the direction whose worst-case resulting candidate
        group (grouped by what the *next* reading would look like) is
        smallest. This mirrors the standard greedy strategy for
        "20 questions"-style identification problems.
        """
        best_dir, best_worst = None, None
        for d in open_dirs:
            groups = {}
            for c in candidates:
                if d not in self.graph.get(c, set()):
                    continue
                nb = neighbor(c, d)
                groups.setdefault(signature(self.graph, nb), []).append(nb)
            if not groups:
                continue
            worst = max(len(g) for g in groups.values())
            if best_worst is None or worst < best_worst:
                best_worst, best_dir = worst, d
        return best_dir

    def localize(self, robot: MazeRobotSim, max_steps=30, verbose=True):
        """
        Returns (localized_cell, num_disambiguation_moves, history) using
        only robot.sense() / robot.step().
        """
        reading = robot.sense()
        candidates = self._candidates_matching(reading)
        history = [(dict(reading), set(candidates))]

        if not candidates:
            raise LocalizationError(
                "Initial sensor reading matches no cell in the map — "
                "the robot may be outside the mapped area, or the map is stale."
            )
        if verbose:
            print(f"Initial reading {reading} -> {len(candidates)} candidate cell(s).")

        moves = 0
        while len(candidates) > 1 and moves < max_steps:
            open_dirs = [d for d, open_ in reading.items() if open_]
            d = self._choose_disambiguating_direction(candidates, open_dirs)
            if d is None:
                # None of the currently-open directions is a valid map edge
                # for any candidate — give up disambiguating and report the
                # (ambiguous) set as-is rather than wandering blindly.
                break
            moved = robot.step(d)
            if not moved:
                # Shouldn't happen since d came from a live open reading,
                # but guard against race conditions / sim edge cases.
                break
            reading = robot.sense()
            new_candidates = set()
            for c in candidates:
                if d in self.graph.get(c, set()):
                    nb = neighbor(c, d)
                    if signature(self.graph, nb) == frozenset(
                        x for x, open_ in reading.items() if open_
                    ):
                        new_candidates.add(nb)
            candidates = new_candidates
            moves += 1
            history.append((dict(reading), set(candidates)))
            if verbose:
                print(f"  Moved {d}; new reading {reading} -> "
                      f"{len(candidates)} candidate(s).")

        if len(candidates) == 1:
            return next(iter(candidates)), moves, history
        elif len(candidates) == 0:
            raise LocalizationError(
                "Candidate set collapsed to zero — sensor readings became "
                "inconsistent with the map."
            )
        else:
            raise LocalizationError(
                f"Could not disambiguate within {max_steps} moves; "
                f"{len(candidates)} candidates remain: {candidates}"
            )


# ============================================================================
# 4. Shortest-path planning — A* over the map graph (uniform edge cost 1,
#    Manhattan-distance heuristic; equivalent to BFS on this unweighted
#    grid but demonstrates the general algorithm).
# ============================================================================

def astar(graph, start, goal):
    if start == goal:
        return [start]

    def h(cell):
        return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])

    open_heap = [(h(start), 0, start)]
    came_from = {}
    g_score = {start: 0}
    visited = set()

    while open_heap:
        _, g, current = heapq.heappop(open_heap)
        if current in visited:
            continue
        visited.add(current)
        if current == goal:
            path = [current]
            while path[-1] in came_from:
                path.append(came_from[path[-1]])
            return list(reversed(path))

        for d in graph.get(current, set()):
            nb = neighbor(current, d)
            if nb not in graph:
                continue
            tentative_g = g + 1
            if tentative_g < g_score.get(nb, float("inf")):
                g_score[nb] = tentative_g
                came_from[nb] = current
                heapq.heappush(open_heap, (tentative_g + h(nb), tentative_g, nb))

    raise ValueError(f"No path found from {start} to {goal} in the map graph.")


def path_to_directions(path):
    dirs = []
    for a, b in zip(path, path[1:]):
        delta = (b[0] - a[0], b[1] - a[1])
        dirs.append(DIR_OF_DELTA[delta])
    return dirs


# ============================================================================
# 5. Plotting
# ============================================================================

def plot_run(graph, rows, cols, cell_size, history, path, robot, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(cols, rows))
    t = 0.06
    cs = cell_size

    def cell_center(r, c):
        return c * cs + cs / 2.0, -(r * cs + cs / 2.0)

    for (r, c), dirs in graph.items():
        cx, cy = cell_center(r, c)
        if N not in dirs and r == 0:
            ax.add_patch(plt.Rectangle((cx - cs/2, cy + cs/2 - t/2), cs, t, color="black"))
        if W not in dirs and c == 0:
            ax.add_patch(plt.Rectangle((cx - cs/2 - t/2, cy - cs/2), t, cs, color="black"))
        if S not in dirs:
            ax.add_patch(plt.Rectangle((cx - cs/2, cy - cs/2 - t/2), cs, t, color="black"))
        if E not in dirs:
            ax.add_patch(plt.Rectangle((cx + cs/2 - t/2, cy - cs/2), t, cs, color="black"))

    # Candidate cells considered at each stage of localization, faded out.
    final_reading_candidates = history[-1][1]
    for i, (_, cands) in enumerate(history):
        alpha = 0.15 + 0.1 * i
        for (r, c) in cands:
            cx, cy = cell_center(r, c)
            ax.add_patch(plt.Rectangle((cx - cs/2, cy - cs/2), cs, cs,
                                        color="tab:blue", alpha=min(alpha, 0.5), zorder=1))

    # Planned + driven path.
    traj = np.array(robot.trajectory_xy)
    ax.plot(traj[:, 0], traj[:, 1], color="tab:red", linewidth=1.5, alpha=0.85,
             label="robot motion", zorder=4)

    if path:
        px = [cell_center(r, c)[0] for r, c in path]
        py = [cell_center(r, c)[1] for r, c in path]
        ax.plot(px, py, color="tab:green", linewidth=2.5, linestyle="--",
                 alpha=0.9, label="planned shortest path", zorder=3)

    loc_cell = next(iter(final_reading_candidates)) if len(final_reading_candidates) == 1 else None
    if loc_cell:
        lx, ly = cell_center(*loc_cell)
        ax.plot(lx, ly, marker="o", color="cyan", markersize=14,
                 markeredgecolor="black", label="localized position", zorder=6)
    if path:
        gx, gy = cell_center(*path[-1])
        ax.plot(gx, gy, marker="*", color="gold", markersize=22,
                 markeredgecolor="black", label="destination", zorder=6)

    ax.set_xlim(-0.5, cols * cs + 0.5)
    ax.set_ylim(-rows * cs - 0.5, 0.5)
    ax.set_aspect("equal")
    ax.set_title("Localization (blue = candidate cells) + shortest path")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# ============================================================================
# main
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", type=str, default="maze.xml")
    ap.add_argument("--map", type=str, default="discovered_map.json",
                     help="Prior map file (from maze_explorer.py or the ground-truth json).")
    ap.add_argument("--spawn-row", type=int, default=None)
    ap.add_argument("--spawn-col", type=int, default=None)
    ap.add_argument("--dest-row", type=int, default=None)
    ap.add_argument("--dest-col", type=int, default=None)
    ap.add_argument("--cell-size", type=float, default=1.0)
    ap.add_argument("--png", type=str, default="localize_and_navigate.png")
    ap.add_argument("--no-render", dest="render", action="store_false",
                     help="Run headless — don't open a live MuJoCo viewer window.")
    ap.add_argument("--no-realtime", dest="realtime", action="store_false",
                     help="Step as fast as possible instead of pacing to real time "
                          "(viewer still updates, just not watchably slow).")
    ap.add_argument("--hold-open", type=float, default=3.0,
                     help="Seconds to keep the viewer window open after finishing.")
    ap.set_defaults(render=True, realtime=True)
    args = ap.parse_args()

    graph, rows, cols = load_map_graph(args.map)

    rng = random.Random()
    spawn_row = args.spawn_row if args.spawn_row is not None else rng.randrange(rows)
    spawn_col = args.spawn_col if args.spawn_col is not None else rng.randrange(cols)

    dest_row, dest_col = args.dest_row, args.dest_col
    if dest_row is None or dest_col is None:
        dest_row = int(input(f"Destination row (0-{rows-1}): "))
        dest_col = int(input(f"Destination col (0-{cols-1}): "))
    destination = (dest_row, dest_col)
    if destination not in graph:
        raise SystemExit(f"Destination {destination} is outside the mapped {rows}x{cols} grid.")

    print(f"Dropping robot at a physically-unknown-to-itself cell "
          f"(sim spawn = ({spawn_row},{spawn_col}), hidden from the algorithm)...")
    robot = MazeRobotSim(args.xml, cell_size=args.cell_size,
                          spawn_row=spawn_row, spawn_col=spawn_col,
                          render=args.render, realtime=args.realtime)

    try:
        print("\n--- Localizing ---")
        localizer = Localizer(graph)
        t0 = time.time()
        localized_cell, disamb_moves, history = localizer.localize(robot)
        print(f"Localized to cell {localized_cell} after {disamb_moves} disambiguation "
              f"move(s) ({time.time()-t0:.2f}s).")
        print(f"(Ground truth, for verification only: {robot.debug_true_cell()} — "
              f"{'MATCH' if robot.debug_true_cell() == localized_cell else 'MISMATCH!'})")

        print(f"\n--- Planning shortest path to destination {destination} ---")
        path = astar(graph, localized_cell, destination)
        directions = path_to_directions(path)
        print(f"Path ({len(path)} cells, {len(directions)} moves): {path}")

        print("\n--- Navigating ---")
        for d in directions:
            ok = robot.step(d)
            if not ok:
                raise SystemExit(f"Navigation failed: sensor says {d} is blocked but "
                                  f"the map said it was open. Map may be stale.")
        print(f"Arrived. Ground-truth final cell: {robot.debug_true_cell()} "
              f"({'SUCCESS' if robot.debug_true_cell() == destination else 'OFF TARGET'})")

        plot_run(graph, rows, cols, args.cell_size, history, path, robot, args.png)
        print(f"\nSaved visualization to {args.png}")
    finally:
        robot.close(hold_open=args.hold_open)


if __name__ == "__main__":
    main()
