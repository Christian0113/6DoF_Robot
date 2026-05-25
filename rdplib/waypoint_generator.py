"""Generate TCP waypoints for writing the character 大 on the base xoy plane."""

from __future__ import annotations

import pathlib

import numpy as np
import yaml


def _lerp3(start: np.ndarray, end: np.ndarray, num_points: int) -> np.ndarray:
    """Equally spaced points on a line segment (including endpoints)."""
    t = np.linspace(0.0, 1.0, num_points)
    return start[None, :] + t[:, None] * (end - start)[None, :]


def generate_da_waypoints(
    center: tuple[float, float] = (0.40, 0.00),
    paper_z: float = 0.05,
    pen_up_z: float = 0.08,
    width: float = 0.09,
    height: float = 0.07,
    points_per_stroke: int = 6,
) -> dict:
    """Build waypoint data for writing 大 on a horizontal sheet on the xoy plane.

    Stroke order (standard): 横 -> 撇 -> 捺
    """
    cx, cy = center
    half_w = width / 2.0
    half_h = height / 2.0
    z_down = paper_z
    z_up = pen_up_z

    # Top horizontal stroke (left -> right)
    heng_start = np.array([cx - half_w, cy + half_h * 0.55, z_down])
    heng_end = np.array([cx + half_w, cy + half_h * 0.55, z_down])
    heng = _lerp3(heng_start, heng_end, points_per_stroke)

    # Left falling stroke 撇 (top-center -> bottom-left)
    pie_start = np.array([cx - half_w * 0.05, cy + half_h * 0.40, z_down])
    pie_end = np.array([cx - half_w * 0.92, cy - half_h, z_down])
    pie = _lerp3(pie_start, pie_end, points_per_stroke)

    # Right falling stroke 捺 (top-center -> bottom-right)
    na_start = np.array([cx + half_w * 0.05, cy + half_h * 0.40, z_down])
    na_end = np.array([cx + half_w * 0.92, cy - half_h, z_down])
    na = _lerp3(na_start, na_end, points_per_stroke)

    def to_list(points: np.ndarray) -> list[list[float]]:
        return [[round(float(value), 4) for value in row] for row in points]

    def pen_up_travel(target_x: float, target_y: float) -> list[list[float]]:
        return [
            [round(float(target_x), 4), round(float(target_y), 4), round(z_up, 4)],
            [round(float(target_x), 4), round(float(target_y), 4), round(z_down, 4)],
        ]

    strokes = [
        {
            "id": 1,
            "name": "横",
            "pen": "down",
            "points": to_list(heng),
        },
        {
            "id": 2,
            "name": "pen_up_to_撇",
            "pen": "up",
            "points": pen_up_travel(pie_start[0], pie_start[1]),
        },
        {
            "id": 3,
            "name": "撇",
            "pen": "down",
            "points": to_list(pie),
        },
        {
            "id": 4,
            "name": "pen_up_to_捺",
            "pen": "up",
            "points": pen_up_travel(na_start[0], na_start[1]),
        },
        {
            "id": 5,
            "name": "捺",
            "pen": "down",
            "points": to_list(na),
        },
    ]

    all_points: list[list[float]] = []
    for stroke in strokes:
        all_points.extend(stroke["points"])

    return {
        "task": "write character 大 on paper in base xoy plane",
        "paper": {
            "plane": "xoy",
            "z": paper_z,
            "z_pen_up": pen_up_z,
            "note": "纸面位于基座坐标系 xoy 平面，z 为纸面高度 [m]",
        },
        "character": {
            "glyph": "大",
            "center": [round(cx, 4), round(cy, 4)],
            "size_m": [round(width, 4), round(height, 4)],
            "stroke_order": ["横", "撇", "捺"],
        },
        "strokes": strokes,
        "waypoints": all_points,
    }


def save_da_waypoints(path: str | pathlib.Path) -> dict:
    """Generate and save waypoint file."""
    data = generate_da_waypoints()
    path = pathlib.Path(path)

    header = (
        "# TCP waypoints for writing 大 on paper (base xoy plane)\n"
        "# Units: meter, frame 0\n"
        "# Format: each row is [x, y, z]; pen-up rows have z = z_pen_up\n"
    )
    with open(path, "w", encoding="utf-8") as file:
        file.write(header)
        yaml.dump(data, file, allow_unicode=True, sort_keys=False, default_flow_style=False)

    csv_path = path.with_suffix(".csv")
    with open(csv_path, "w", encoding="utf-8") as file:
        file.write("x\ty\tz\tstroke\tpen\n")
        for stroke in data["strokes"]:
            for point in stroke["points"]:
                file.write(
                    f"{point[0]:.4f}\t{point[1]:.4f}\t{point[2]:.4f}\t"
                    f"{stroke['name']}\t{stroke['pen']}\n"
                )

    return data


def load_stroke_segments(path: str | pathlib.Path) -> list[dict]:
    """Load stroke groups with 3xN point arrays from a waypoint yaml file."""
    with open(path, "r", encoding="utf-8") as file:
        data = yaml.load(file, Loader=yaml.loader.SafeLoader)

    segments: list[dict] = []
    for stroke in data.get("strokes", []):
        points = np.array(stroke["points"], dtype=float).T
        segments.append(
            {
                "name": stroke["name"],
                "pen": stroke["pen"],
                "points": points,
            }
        )
    return segments


def load_waypoints(path: str | pathlib.Path) -> np.ndarray:
    """Load flat TCP waypoint array (3, N) from a waypoint yaml file."""
    with open(path, "r", encoding="utf-8") as file:
        data = yaml.load(file, Loader=yaml.loader.SafeLoader)

    points = data.get("waypoints")
    if points is None:
        points = []
        for stroke in data.get("strokes", []):
            points.extend(stroke["points"])

    return np.array(points, dtype=float).T


if __name__ == "__main__":
    output = pathlib.Path(__file__).resolve().parent.parent / "waypoints_da.yaml"
    data = save_da_waypoints(output)
    print(f"Saved {len(data['waypoints'])} waypoints to {output}")
    print(f"Saved CSV copy to {output.with_suffix('.csv')}")
