"""
run_demo.py
-----------
One-shot demo: generates a maze, drops the robot at a start cell of your
choosing (or a random one), fully explores it, and saves/plots the result.

    python3 run_demo.py --rows 10 --cols 10 --seed 3 --random-start
"""
import argparse
import random

from maze_generator import generate_maze, build_mjcf, maze_to_json
from maze_explorer import MazeExplorer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--start-row", type=int, default=0)
    ap.add_argument("--start-col", type=int, default=0)
    ap.add_argument("--random-start", action="store_true",
                     help="Ignore --start-row/--start-col and drop the robot on a random cell.")
    ap.add_argument("--xml-out", type=str, default="maze.xml")
    ap.add_argument("--map-out", type=str, default="discovered_map.json")
    ap.add_argument("--png-out", type=str, default="discovered_map.png")
    args = ap.parse_args()

    # 1. Generate the maze and write the MuJoCo model.
    grid = generate_maze(args.rows, args.cols, seed=args.seed)
    xml = build_mjcf(grid, args.rows, args.cols)
    with open(args.xml_out, "w") as f:
        f.write(xml)
    maze_to_json(grid, args.rows, args.cols, args.xml_out.replace(".xml", "_ground_truth.json"))
    print(f"[1/2] Maze generated -> {args.xml_out}  ({args.rows}x{args.cols})")

    # 2. Place the bot and let it fully explore + map the maze.
    if args.random_start:
        rng = random.Random(args.seed)
        start_row = rng.randrange(args.rows)
        start_col = rng.randrange(args.cols)
    else:
        start_row, start_col = args.start_row, args.start_col

    print(f"[2/2] Dropping robot at cell ({start_row}, {start_col}) and exploring...")
    explorer = MazeExplorer(args.xml_out, args.rows, args.cols,
                             start_row=start_row, start_col=start_col)
    stats = explorer.explore()
    explorer.save_map(args.map_out)
    explorer.plot_map(args.png_out)

    print(f"Map saved to {args.map_out}")
    print(f"Map image saved to {args.png_out}")
    print(stats)


if __name__ == "__main__":
    main()
