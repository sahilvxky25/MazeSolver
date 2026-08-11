"""
maze_explorer.py
-----------------
Drops a robot anywhere inside a MuJoCo maze (built by maze_generator.py) and
drives it to completely traverse the maze, using ONLY its own onboard
rangefinder sensors (never the ground-truth maze file) to sense walls as it
goes. While exploring, it builds an occupancy map of every wall it detected
and saves that discovered map to disk (JSON + PNG).

Algorithm: right-hand wall following ("right-hand rule").
Because maze_generator.py always builds a *perfect* maze (a spanning tree —
fully connected, zero loops), a wall-following robot is mathematically
guaranteed to visit every single cell and eventually return to its start,
regardless of which cell it starts in. This is a classical, well-known
result for simply-connected mazes, and it lets us do full coverage with a
purely local, memoryless control policy (no path planning / global map
needed to decide where to go next) — the map is a byproduct we record for
later, not something the controller consults to move.

Usage:
    python3 maze_explorer.py --xml maze.xml --rows 6 --cols 6 \\
        --start-row 3 --start-col 4 --out discovered_map.json
"""

import argparse
import json
import time
import numpy as np
import mujoco

from maze_viewer import LiveViewer

DIR_VEC = {"N": (0.0, 1.0), "S": (0.0, -1.0), "E": (1.0, 0.0), "W": (-1.0, 0.0)}
OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}
CLOCKWISE = {"N": "E", "E": "S", "S": "W", "W": "N"}
COUNTERCLOCKWISE = {"N": "W", "W": "S", "S": "E", "E": "N"}
SENSOR_NAME = {"N": "rf_north", "S": "rf_south", "E": "rf_east", "W": "rf_west"}


class MazeExplorer:
    def __init__(self, xml_path, rows, cols, cell_size=1.0,
                 start_row=0, start_col=0, open_threshold_frac=0.6,
                 render=True, realtime=True):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.rows = rows
        self.cols = cols
        self.cell_size = cell_size

        # Live MuJoCo viewer — makes the robot's exploration visible in real
        # time instead of the sim just running silently in the background.
        self.live = LiveViewer(self.model, self.data, enabled=render, realtime=realtime)

        cutoff = self.model.sensor_cutoff[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "rf_east")
        ]
        self.open_threshold = cutoff * open_threshold_frac

        # IMPORTANT: the robot's slide-joint qpos is a DISPLACEMENT added on
        # top of the "robot" body's baked <body pos="..."> from the XML —
        # it is NOT an absolute world coordinate. We read that baked offset
        # once here so every world<->qpos conversion below is correct
        # regardless of which start cell maze_generator.py baked in.
        self.body_base = self.model.body("robot").pos[:2].copy()

        # Drop the robot at the requested start cell — "anywhere in the map" —
        # overriding whatever position was baked into the XML at generation time.
        self.set_position(start_row, start_col)
        self.current_cell = (start_row, start_col)
        self.start_cell = (start_row, start_col)
        self.live.sync(self.model, self.data)  # show the starting pose right away

        self.sensed_walls = {}     # {(r,c): {'N':bool_open, 'E':..., 'S':..., 'W':...}}
        self.visited = set()
        self.path_cells = []       # ordered list of visited cells (with repeats)
        self.trajectory_xy = []    # continuous (x,y) samples for plotting/animation
        self.total_sim_steps = 0

    # -- geometry helpers -------------------------------------------------
    def cell_center(self, r, c):
        x = c * self.cell_size + self.cell_size / 2.0
        y = -(r * self.cell_size + self.cell_size / 2.0)
        return np.array([x, y])

    def world_pos(self):
        """Absolute world (x, y) of the robot right now."""
        return self.data.qpos[:2] + self.body_base

    def set_position(self, r, c):
        x, y = self.cell_center(r, c)
        self.data.qpos[:2] = np.array([x, y]) - self.body_base
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

    def cell_of(self, r, c, direction):
        dr = {"N": -1, "S": 1, "E": 0, "W": 0}[direction]
        dc = {"N": 0, "S": 0, "E": 1, "W": -1}[direction]
        return (r + dr, c + dc)

    def in_bounds(self, cell):
        r, c = cell
        return 0 <= r < self.rows and 0 <= c < self.cols

    # -- sensing ------------------------------------------------------------
    def sense_walls(self):
        """Reads the 4 onboard rangefinders and returns {'N':open?, ...}."""
        reading = {}
        for d, sname in SENSOR_NAME.items():
            val = self.data.sensor(sname).data[0]
            reading[d] = bool(val > self.open_threshold)
        return reading

    # -- motion -------------------------------------------------------------
    def move_to(self, target_xy, kp=6.0, vmax=2.5, tol=0.02, max_steps=4000):
        act_x = self.model.actuator("act_x").id
        act_y = self.model.actuator("act_y").id
        reached = False
        for _ in range(max_steps):
            pos = self.world_pos()
            err = target_xy - pos
            dist = np.linalg.norm(err)
            if dist < tol:
                reached = True
                break
            vel = np.clip(kp * err, -vmax, vmax)
            self.data.ctrl[act_x] = vel[0]
            self.data.ctrl[act_y] = vel[1]
            mujoco.mj_step(self.model, self.data)
            self.live.sync(self.model, self.data)  # render this physics step
            self.total_sim_steps += 1
            self.trajectory_xy.append(pos.copy())
        # Stop cleanly at the target.
        self.data.ctrl[act_x] = 0.0
        self.data.ctrl[act_y] = 0.0
        self.data.qvel[:2] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.live.sync(self.model, self.data)
        return reached

    def close(self, hold_open=3.0):
        """Close the live viewer window (optionally holding the last frame)."""
        self.live.close(hold_open=hold_open)

    # -- main exploration loop ----------------------------------------------
    def explore(self, max_moves=None, verbose=True):
        total_cells = self.rows * self.cols
        if max_moves is None:
            # Safe upper bound: a wall follower crosses each of the
            # (total_cells - 1) tree edges at most twice; pad generously.
            max_moves = total_cells * 6 + 20

        facing = "N"
        moves = 0
        # Record the very first sensor reading before any movement.
        self._record_current_cell()

        while moves < max_moves:
            fully_covered = len(self.visited) == total_cells
            if fully_covered and self.current_cell == self.start_cell and moves > 0:
                break

            reading = self.sense_walls()
            # Right-hand rule priority: right, straight, left, back.
            order = [CLOCKWISE[facing], facing, COUNTERCLOCKWISE[facing], OPPOSITE[facing]]
            next_dir = next((d for d in order if reading[d]), None)

            if next_dir is None:
                # Fully walled in (shouldn't happen in a connected maze) — stop.
                if verbose:
                    print("No open direction found — stopping.")
                break

            target_cell = self.cell_of(*self.current_cell, next_dir)
            if not self.in_bounds(target_cell):
                if verbose:
                    print(f"Sensor suggested a move outside the grid "
                          f"({self.current_cell} -> {target_cell}); stopping.")
                break

            target_xy = self.cell_center(*target_cell)
            reached = self.move_to(target_xy)
            if not reached and verbose:
                print(f"Warning: did not cleanly reach {target_cell} "
                      f"from {self.current_cell} within the step budget.")
            self.current_cell = target_cell
            facing = next_dir
            moves += 1
            self._record_current_cell()

        total_visited = len(self.visited)
        if verbose:
            print(f"Exploration finished after {moves} moves "
                  f"({self.total_sim_steps} sim steps).")
            print(f"Cells visited: {total_visited}/{total_cells} "
                  f"({'FULL COVERAGE' if total_visited == total_cells else 'INCOMPLETE'})")

        return {
            "moves": moves,
            "sim_steps": self.total_sim_steps,
            "cells_visited": total_visited,
            "total_cells": total_cells,
            "full_coverage": total_visited == total_cells,
        }

    def _record_current_cell(self):
        cell = self.current_cell
        if cell not in self.sensed_walls:
            self.sensed_walls[cell] = self.sense_walls()
        self.visited.add(cell)
        self.path_cells.append(cell)
        self.trajectory_xy.append(self.world_pos().copy())

    # -- output ---------------------------------------------------------
    def save_map(self, out_json_path):
        serializable = {
            f"{r},{c}": {d: bool(open_) for d, open_ in walls.items()}
            for (r, c), walls in self.sensed_walls.items()
        }
        payload = {
            "rows": self.rows,
            "cols": self.cols,
            "cell_size": self.cell_size,
            "start_cell": list(self.start_cell),
            "cells_visited": len(self.visited),
            "total_cells": self.rows * self.cols,
            "path_cells": [list(c) for c in self.path_cells],
            "walls": serializable,
        }
        with open(out_json_path, "w") as f:
            json.dump(payload, f, indent=2)
        return payload

    def plot_map(self, out_png_path, ground_truth_json=None):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(self.cols, self.rows))
        t = 0.06
        cs = self.cell_size

        for (r, c), walls in self.sensed_walls.items():
            cx = c * cs + cs / 2.0
            cy = -(r * cs + cs / 2.0)
            if not walls.get("N", True) and r == 0:
                ax.add_patch(plt.Rectangle((cx - cs/2, cy + cs/2 - t/2), cs, t, color="black"))
            if not walls.get("W", True) and c == 0:
                ax.add_patch(plt.Rectangle((cx - cs/2 - t/2, cy - cs/2), t, cs, color="black"))
            if not walls.get("S", True):
                ax.add_patch(plt.Rectangle((cx - cs/2, cy - cs/2 - t/2), cs, t, color="black"))
            if not walls.get("E", True):
                ax.add_patch(plt.Rectangle((cx + cs/2 - t/2, cy - cs/2), t, cs, color="black"))

        traj = np.array(self.trajectory_xy)
        if len(traj):
            ax.plot(traj[:, 0], traj[:, 1], color="tab:red", linewidth=1.5, alpha=0.8,
                     label="robot path")
        sx, sy = self.cell_center(*self.start_cell)
        ax.plot(sx, sy, marker="*", color="gold", markersize=20,
                 markeredgecolor="black", label="start", zorder=5)

        ax.set_xlim(-0.5, self.cols * cs + 0.5)
        ax.set_ylim(-self.rows * cs - 0.5, 0.5)
        ax.set_aspect("equal")
        ax.set_title(f"Discovered maze map — {len(self.visited)}/{self.rows*self.cols} cells")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(out_png_path, dpi=150)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Fully traverse a MuJoCo maze and save the discovered map.")
    ap.add_argument("--xml", type=str, default="maze.xml")
    ap.add_argument("--rows", type=int, required=True)
    ap.add_argument("--cols", type=int, required=True)
    ap.add_argument("--cell-size", type=float, default=1.0)
    ap.add_argument("--start-row", type=int, default=0)
    ap.add_argument("--start-col", type=int, default=0)
    ap.add_argument("--out", type=str, default="discovered_map.json")
    ap.add_argument("--png", type=str, default="discovered_map.png")
    ap.add_argument("--no-render", dest="render", action="store_false",
                     help="Run headless — don't open a live MuJoCo viewer window.")
    ap.add_argument("--no-realtime", dest="realtime", action="store_false",
                     help="Step as fast as possible instead of pacing to real time "
                          "(viewer still updates, just not watchably slow).")
    ap.add_argument("--hold-open", type=float, default=3.0,
                     help="Seconds to keep the viewer window open after finishing.")
    ap.set_defaults(render=True, realtime=True)
    args = ap.parse_args()

    explorer = MazeExplorer(args.xml, args.rows, args.cols, cell_size=args.cell_size,
                             start_row=args.start_row, start_col=args.start_col,
                             render=args.render, realtime=args.realtime)

    try:
        t0 = time.time()
        stats = explorer.explore()
        dt = time.time() - t0

        explorer.save_map(args.out)
        explorer.plot_map(args.png)

        print(f"Wall-clock exploration time: {dt:.2f}s")
        print(f"Saved discovered map to {args.out}")
        print(f"Saved map image to {args.png}")
        print(json.dumps(stats, indent=2))
    finally:
        explorer.close(hold_open=args.hold_open)


if __name__ == "__main__":
    main()
