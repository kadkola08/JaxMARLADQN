"""
Box Pushing Environment Visualizer.

Renders BoxPushingSimple (and compatible) state as a grid: agents (directed),
small/large boxes, goal zone. Follows the same tile-based pattern as
OvercookedVisualizer using jaxmarl.viz.grid_rendering.
"""

import math
import numpy as np
from functools import partial

from jaxmarl.viz.window import Window
import jaxmarl.viz.grid_rendering as rendering

# Cell types for the grid
EMPTY = 0
GOAL = 1
AGENT = 2
SMALL_BOX = 3
LARGE_BOX = 4

TILE_PIXELS = 32

# RGB colors for tiles (numpy uint8)
COLORS = {
    "red": np.array([255, 0, 0], dtype=np.uint8),
    "blue": np.array([0, 0, 255], dtype=np.uint8),
    "green": np.array([0, 200, 0], dtype=np.uint8),
    "grey": np.array([100, 100, 100], dtype=np.uint8),
    "white": np.array([255, 255, 255], dtype=np.uint8),
    "black": np.array([25, 25, 25], dtype=np.uint8),
    "brown": np.array([139, 90, 43], dtype=np.uint8),
    "yellow": np.array([255, 220, 0], dtype=np.uint8),
    "light_green": np.array([144, 238, 144], dtype=np.uint8),
}

# Agent index -> color name
AGENT_COLORS = ["red", "blue"]


def _state_to_grid(state, grid_size, num_small_boxes=0, num_large_boxes=1):
    """
    Convert box pushing state to a grid of (type, color_idx, extra).
    type: EMPTY, GOAL, AGENT, SMALL_BOX, LARGE_BOX
    color_idx: for agents 0 or 1; unused otherwise
    extra: for agents, orientation 0..3

    Env coords: x column, y row with y=grid_size-1 = top (goal). We store grid
    with row 0 = top of image (goal row), so display row = grid_size - 1 - y.
    """
    agent_pos = np.asarray(state.agent_pos)
    agent_orient = np.asarray(state.agent_orient)
    small_box_pos = np.asarray(state.small_box_pos)
    large_box_pos = np.asarray(state.large_box_pos)

    # (grid_size, grid_size, 3): type, color_idx, extra. Row 0 = top of image = goal row.
    grid = np.zeros((grid_size, grid_size, 3), dtype=np.int32)
    grid[:, :, 0] = EMPTY

    def row(y):
        return int(grid_size - 1 - y)

    # Goal row at top of image (env y = grid_size - 1 -> row 0)
    grid[0, :, 0] = GOAL

    # Large box cells (overwrite goal/empty)
    for b in range(num_large_boxes):
        cells = large_box_pos[b]
        for c in range(cells.shape[0]):
            x, y = int(cells[c, 0]), int(cells[c, 1])
            if 0 <= x < grid_size and 0 <= y < grid_size:
                grid[row(y), x, 0] = LARGE_BOX

    # Small box cells
    for b in range(num_small_boxes):
        x, y = int(small_box_pos[b, 0]), int(small_box_pos[b, 1])
        if 0 <= x < grid_size and 0 <= y < grid_size:
            grid[row(y), x, 0] = SMALL_BOX

    # Agents (drawn last so they are on top)
    num_agents = agent_pos.shape[0]
    for i in range(num_agents):
        x, y = int(agent_pos[i, 0]), int(agent_pos[i, 1])
        if 0 <= x < grid_size and 0 <= y < grid_size:
            grid[row(y), x, 0] = AGENT
            grid[row(y), x, 1] = i  # color_idx
            grid[row(y), x, 2] = int(agent_orient[i])  # orientation

    return grid


class BoxPushingVisualizer:
    """
    Renders BoxPushingSimple (and compatible) states in a window or to GIF.
    """

    tile_cache = {}

    def __init__(self, tile_size=TILE_PIXELS):
        self.window = None
        self.tile_size = tile_size

    def _lazy_init_window(self):
        if self.window is None:
            self.window = Window("Box Pushing")

    def show(self, block=False):
        self._lazy_init_window()
        self.window.show(block=block)

    def render(
        self,
        state,
        grid_size,
        num_small_boxes=0,
        num_large_boxes=1,
        highlight=True,
    ):
        """Render a single state in the window."""
        self._lazy_init_window()
        grid = _state_to_grid(state, grid_size, num_small_boxes, num_large_boxes)
        highlight_mask = np.zeros((grid_size, grid_size), dtype=bool) if highlight else None
        img = self._render_grid(grid, highlight_mask=highlight_mask)
        self.window.show_img(img)

    def animate(
        self,
        state_list,
        grid_size,
        num_small_boxes=0,
        num_large_boxes=1,
        filename="box_pushing.gif",
        duration=0.5,
    ):
        """Render a list of states to a GIF file."""
        import imageio

        frames = []
        for state in state_list:
            grid = _state_to_grid(state, grid_size, num_small_boxes, num_large_boxes)
            frame = self._render_grid(grid, highlight_mask=None)
            frames.append(frame)
        imageio.mimsave(filename, frames, "GIF", duration=duration)

    @classmethod
    def _render_tile(
        cls,
        cell_type,
        color_idx,
        extra,
        highlight=False,
        tile_size=TILE_PIXELS,
        subdivs=3,
    ):
        """Render one tile and optionally use cache."""
        key = (cell_type, color_idx, extra, highlight, tile_size)
        if key in cls.tile_cache:
            return cls.tile_cache[key]

        img = np.zeros((tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8)

        # Grid lines
        rendering.fill_coords(img, rendering.point_in_rect(0, 0.031, 0, 1), COLORS["grey"])
        rendering.fill_coords(img, rendering.point_in_rect(0, 1, 0, 0.031), COLORS["grey"])

        if cell_type == EMPTY:
            rendering.fill_coords(img, rendering.point_in_rect(0, 1, 0, 1), COLORS["white"])
        elif cell_type == GOAL:
            rendering.fill_coords(img, rendering.point_in_rect(0, 1, 0, 1), COLORS["grey"])
            rendering.fill_coords(img, rendering.point_in_rect(0.1, 0.9, 0.1, 0.9), COLORS["light_green"])
        elif cell_type == AGENT:
            # Fill tile background (floor) so no black around the triangle
            rendering.fill_coords(img, rendering.point_in_rect(0, 1, 0, 1), COLORS["white"])
            # Agent: triangle nose = direction of movement (forward). Env: 0=right, 1=up, 2=left, 3=down.
            # Triangle with point on LEFT (base right); theta = pi - 0.5*pi*extra so nose points
            # 0->right, 1->up, 2->left, 3->down (matches env movement direction).
            color_name = AGENT_COLORS[color_idx] if color_idx < len(AGENT_COLORS) else "green"
            tri_fn = rendering.point_in_triangle(
                (0.87, 0.19),   # base right
                (0.13, 0.50),   # point (nose) left
                (0.87, 0.81),   # base right
            )
            theta = math.pi - 0.5 * math.pi * extra
            tri_fn = rendering.rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=theta)
            rendering.fill_coords(img, tri_fn, COLORS[color_name])
        elif cell_type == SMALL_BOX:
            # Fill tile background first so no black border
            rendering.fill_coords(img, rendering.point_in_rect(0, 1, 0, 1), COLORS["white"])
            rendering.fill_coords(img, rendering.point_in_rect(0.1, 0.9, 0.1, 0.9), COLORS["brown"])
        elif cell_type == LARGE_BOX:
            rendering.fill_coords(img, rendering.point_in_rect(0, 1, 0, 1), COLORS["yellow"])

        if highlight:
            rendering.highlight_img(img)

        img = rendering.downsample(img, subdivs)
        cls.tile_cache[key] = img
        return img

    @classmethod
    def _render_grid(
        cls,
        grid,
        tile_size=TILE_PIXELS,
        highlight_mask=None,
    ):
        """Render full grid to an image. grid shape (H, W, 3) with (type, color_idx, extra)."""
        if highlight_mask is None:
            highlight_mask = np.zeros((grid.shape[0], grid.shape[1]), dtype=bool)

        height_px = grid.shape[0] * tile_size
        width_px = grid.shape[1] * tile_size
        img = np.zeros((height_px, width_px, 3), dtype=np.uint8)

        for y in range(grid.shape[0]):
            for x in range(grid.shape[1]):
                cell_type = grid[y, x, 0]
                color_idx = grid[y, x, 1]
                extra = grid[y, x, 2]
                tile_img = cls._render_tile(
                    cell_type,
                    color_idx,
                    extra,
                    highlight=highlight_mask[y, x],
                    tile_size=tile_size,
                )
                ymin = y * tile_size
                ymax = (y + 1) * tile_size
                xmin = x * tile_size
                xmax = (x + 1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img

        return img

    def close(self):
        if self.window is not None:
            self.window.close()
            self.window = None