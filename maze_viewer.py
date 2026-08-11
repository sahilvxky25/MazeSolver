"""
maze_viewer.py
---------------
Small shared helper that opens a live MuJoCo viewer window (the "passive"
viewer API) and paces physics stepping to roughly real time, so you can
actually watch the robot move instead of the simulation running silently in
the background and only writing out a PNG/JSON at the end.

Both maze_explorer.py and maze_localize_and_navigate.py use it the same way:

    live = LiveViewer(model, data, enabled=True)
    ...
    mujoco.mj_step(model, data)
    live.sync(model, data)     # call after every single mj_step
    ...
    live.close(hold_open=3.0)  # keep the window up for a few seconds at the end

If no display is available (headless server/container, SSH without X
forwarding, etc.), opening the window fails once, LiveViewer prints a single
warning, and silently becomes a no-op from then on so the script still runs
(just without a visible robot) instead of crashing.
"""

import os
import platform
import sys
import time
import mujoco
import mujoco.viewer


def _display_available():
    """
    Best-effort check for whether a real display exists. This matters
    because on Linux, MuJoCo's GLFW backend hard-exits the whole process
    (below the Python exception layer, so try/except can't catch it) if it
    can't open a window — so we must avoid even attempting launch_passive()
    when there's clearly no display, rather than relying on catching a
    failure afterwards.
    """
    system = platform.system()
    if system == "Linux":
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True  # macOS/Windows: assume yes; try/except below is the fallback


class LiveViewer:
    def __init__(self, model, data, enabled=True, realtime=True):
        self.enabled = enabled
        self.realtime = realtime
        self.handle = None
        self._sim_t0 = None
        self._wall_t0 = None

        if not self.enabled:
            return

        if not _display_available():
            print(
                "[maze_viewer] No display detected (DISPLAY/WAYLAND_DISPLAY not "
                "set) — continuing headless. Run this on a machine with a "
                "display (or via X11 forwarding/VNC) to watch the robot move live."
            )
            self.enabled = False
            return

        if platform.system() == "Darwin" and "mjpython" not in os.path.basename(sys.executable):
            print(
                "[maze_viewer] On macOS the live viewer must be launched with "
                "`mjpython`, not plain `python3` — continuing headless. "
                "Re-run with e.g. `mjpython maze_explorer.py ...` to watch it live."
            )
            self.enabled = False
            return

        try:
            self.handle = mujoco.viewer.launch_passive(model, data)
        except Exception as exc:
            print(
                f"[maze_viewer] Could not open a live MuJoCo viewer window "
                f"({exc!r}) — continuing headless."
            )
            self.handle = None
            self.enabled = False

    def sync(self, model, data):
        """Call once right after every mujoco.mj_step(model, data)."""
        if not self.handle:
            return
        if not self.handle.is_running():
            # The user closed the window — stop trying to render, but let
            # the simulation keep running headless rather than crashing.
            self.handle = None
            return

        if self.realtime:
            sim_t = data.time
            now = time.perf_counter()
            if self._sim_t0 is None:
                self._sim_t0, self._wall_t0 = sim_t, now
            target_wall = self._wall_t0 + (sim_t - self._sim_t0)
            lag = target_wall - now
            if 0 < lag < 0.25:  # cap so a slow machine can't stall indefinitely
                time.sleep(lag)

        self.handle.sync()

    def is_open(self):
        return self.handle is not None and self.handle.is_running()

    def close(self, hold_open=0.0):
        """Optionally pause with the final frame on-screen, then close."""
        if self.handle:
            if hold_open > 0:
                time.sleep(hold_open)
            self.handle.close()
            self.handle = None
