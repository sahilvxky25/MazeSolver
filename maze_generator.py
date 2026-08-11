"""
maze_generator.py
------------------
Generates a random "perfect" maze (a spanning tree — exactly one path between
any two cells, no loops) using randomized depth-first search / recursive
backtracking, then exports it as a MuJoCo MJCF (.xml) model that can be
opened directly in the MuJoCo viewer.

The maze is grid-aligned to the world X/Y axes. Each cell is a CELL_SIZE x
CELL_SIZE square. Walls are thin boxes placed on cell boundaries.

The exported model also contains a simple holonomic robot ("robot" body)
that can slide freely in X and Y (two slide joints, velocity-actuated) and
carries four rangefinder sensors pointing along +X, -X, +Y, -Y. Because the
maze is axis-aligned, these four sensors are exactly what's needed to detect
the N/E/S/W walls of whichever cell the robot currently occupies — no
heading/orientation tracking required. This is used by maze_explorer.py.

Usage:
    python3 maze_generator.py --rows 8 --cols 8 --seed 1 --out maze.xml
"""

import argparse
import random
import json

# Directions: (row_delta, col_delta, wall_name, opposite_wall_name)
N, S, E, W = "N", "S", "E", "W"
OPPOSITE = {N: S, S: N, E: W, W: E}
DELTA = {N: (-1, 0), S: (1, 0), E: (0, 1), W: (0, -1)}


def generate_maze(rows: int, cols: int, seed: int = None):
    """
    Returns a dict-of-dicts grid where grid[(r, c)] is a set of directions
    ('N','S','E','W') that are OPEN (i.e. no wall / passage exists) from
    that cell. Any direction not in the set has a wall.

    Generated with randomized recursive backtracking, which always produces
    a "perfect" maze: a spanning tree over the grid cells (fully connected,
    zero loops). This property is important — it's what guarantees a simple
    wall-following robot can traverse 100% of the maze (see maze_explorer.py).
    """
    rng = random.Random(seed)
    grid = {(r, c): set() for r in range(rows) for c in range(cols)}
    visited = set()

    def neighbors(cell):
        r, c = cell
        result = []
        for d, (dr, dc) in DELTA.items():
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                result.append((d, (nr, nc)))
        rng.shuffle(result)
        return result

    start = (0, 0)
    stack = [start]
    visited.add(start)
    while stack:
        current = stack[-1]
        unvisited_neighbors = [
            (d, nb) for d, nb in neighbors(current) if nb not in visited
        ]
        if not unvisited_neighbors:
            stack.pop()
            continue
        d, nxt = unvisited_neighbors[0]
        grid[current].add(d)
        grid[nxt].add(OPPOSITE[d])
        visited.add(nxt)
        stack.append(nxt)

    return grid


def maze_to_json(grid, rows, cols, path):
    """Save the ground-truth maze connectivity to JSON (for reference/debug)."""
    serializable = {f"{r},{c}": sorted(list(grid[(r, c)]))
                     for r in range(rows) for c in range(cols)}
    with open(path, "w") as f:
        json.dump({"rows": rows, "cols": cols, "grid": serializable}, f, indent=2)


def build_mjcf(grid, rows, cols, cell_size=1.0, wall_height=0.5,
               wall_thickness=0.08, start_cell=(0, 0)):
    """
    Converts the maze grid into a MuJoCo MJCF XML string.

    Layout convention: cell (r, c) occupies the square spanning
      x in [c*cell_size, (c+1)*cell_size]
      y in [-(r+1)*cell_size, -r*cell_size]
    i.e. row 0 is at the top (largest y), increasing row goes toward -y,
    increasing col goes toward +x. Cell *centers* are used for wall/robot
    placement math below.

    A wall is drawn once per shared edge (checked from the lower-index cell
    to avoid duplicate overlapping geoms).
    """
    t = wall_thickness
    h = wall_height
    half_h = h / 2.0

    def cell_center(r, c):
        x = c * cell_size + cell_size / 2.0
        y = -(r * cell_size + cell_size / 2.0)
        return x, y

    walls_xml = []

    def add_wall(x, y, size_x, size_y, name):
        walls_xml.append(
            f'      <geom name="{name}" type="box" pos="{x:.4f} {y:.4f} {half_h:.4f}" '
            f'size="{size_x:.4f} {size_y:.4f} {half_h:.4f}" rgba="0.55 0.55 0.6 1" '
            f'class="wall"/>'
        )

    wid = 0
    # Interior + boundary walls, one pass per cell checking S and E (plus
    # global N boundary for row 0 and W boundary for col 0) avoids duplicates.
    for r in range(rows):
        for c in range(cols):
            cx, cy = cell_center(r, c)
            # North boundary (only for top row)
            if r == 0 and N not in grid[(r, c)]:
                add_wall(cx, cy + cell_size / 2.0, cell_size / 2.0 + t, t, f"wall_{wid}"); wid += 1
            # West boundary (only for left col)
            if c == 0 and W not in grid[(r, c)]:
                add_wall(cx - cell_size / 2.0, cy, t, cell_size / 2.0 + t, f"wall_{wid}"); wid += 1
            # South wall (shared with row r+1) — always owned by this cell
            if S not in grid[(r, c)]:
                add_wall(cx, cy - cell_size / 2.0, cell_size / 2.0 + t, t, f"wall_{wid}"); wid += 1
            # East wall (shared with col c+1) — always owned by this cell
            if E not in grid[(r, c)]:
                add_wall(cx + cell_size / 2.0, cy, t, cell_size / 2.0 + t, f"wall_{wid}"); wid += 1

    walls_block = "\n".join(walls_xml)

    total_w = cols * cell_size
    total_h = rows * cell_size
    floor_cx = total_w / 2.0
    floor_cy = -total_h / 2.0

    sx, sy = cell_center(*start_cell)
    sensor_range = cell_size * 0.85  # a bit less than one cell so it reads the near wall

    xml = f"""<mujoco model="maze">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.005" gravity="0 0 -9.81"/>

  <default>
    <default class="wall">
      <geom type="box" contype="1" conaffinity="1" friction="0.8 0.01 0.01"/>
    </default>
  </default>

  <asset>
    <texture type="2d" name="grid" builtin="checker" rgb1="0.2 0.3 0.4" rgb2="0.3 0.4 0.5"
             width="300" height="300"/>
    <material name="grid_mat" texture="grid" texrepeat="{cols} {rows}" reflectance="0.1"/>
  </asset>

  <worldbody>
    <light pos="{floor_cx} {floor_cy} 4" dir="0 0 -1" diffuse="1 1 1" directional="true"/>
    <geom name="floor" type="plane" pos="{floor_cx:.4f} {floor_cy:.4f} 0"
          size="{total_w/2.0+1:.4f} {total_h/2.0+1:.4f} 0.1" material="grid_mat"/>

{walls_block}

    <body name="robot" pos="{sx:.4f} {sy:.4f} 0.15">
      <joint name="robot_x" type="slide" axis="1 0 0" damping="8" limited="false"/>
      <joint name="robot_y" type="slide" axis="0 1 0" damping="8" limited="false"/>
      <geom name="robot_geom" type="cylinder" size="0.15 0.1" rgba="0.9 0.2 0.2 1" mass="1.0"/>
      <!-- Rangefinder sensors fire along each site's local +Z axis, so each
           site below is rotated (axisangle) to point its local Z outward
           in the world direction named. -->
      <site name="s_east"  pos="0.15 0 0"  axisangle="0 1 0 1.5708"  size="0.02"/>
      <site name="s_west"  pos="-0.15 0 0" axisangle="0 1 0 -1.5708" size="0.02"/>
      <site name="s_north" pos="0 0.15 0"  axisangle="1 0 0 -1.5708" size="0.02"/>
      <site name="s_south" pos="0 -0.15 0" axisangle="1 0 0 1.5708"  size="0.02"/>
    </body>
  </worldbody>

  <actuator>
    <velocity name="act_x" joint="robot_x" kv="20" ctrlrange="-3 3"/>
    <velocity name="act_y" joint="robot_y" kv="20" ctrlrange="-3 3"/>
  </actuator>

  <sensor>
    <rangefinder name="rf_east"  site="s_east"  cutoff="{sensor_range:.4f}"/>
    <rangefinder name="rf_west"  site="s_west"  cutoff="{sensor_range:.4f}"/>
    <rangefinder name="rf_north" site="s_north" cutoff="{sensor_range:.4f}"/>
    <rangefinder name="rf_south" site="s_south" cutoff="{sensor_range:.4f}"/>
  </sensor>
</mujoco>
"""
    return xml


def main():
    ap = argparse.ArgumentParser(description="Generate a maze as a MuJoCo MJCF model.")
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--cell-size", type=float, default=1.0)
    ap.add_argument("--out", type=str, default="maze.xml")
    ap.add_argument("--meta-out", type=str, default="maze_ground_truth.json")
    args = ap.parse_args()

    grid = generate_maze(args.rows, args.cols, seed=args.seed)
    xml = build_mjcf(grid, args.rows, args.cols, cell_size=args.cell_size)

    with open(args.out, "w") as f:
        f.write(xml)
    maze_to_json(grid, args.rows, args.cols, args.meta_out)

    print(f"Wrote MuJoCo model to {args.out}")
    print(f"Wrote ground-truth maze connectivity to {args.meta_out}")
    print(f"Grid: {args.rows} rows x {args.cols} cols, cell_size={args.cell_size}")
    print("View it with:  python3 -m mujoco.viewer --mjcf=" + args.out)


if __name__ == "__main__":
    main()
