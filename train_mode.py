import copy
import pickle
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from Astar import astar, dynamic_astar
from BFS import bfs, dynamic_bfs
from GBFS import gbfs, dynamic_gbfs


Grid = List[List[str]]
Point = Tuple[int, int]


def random_grid(rows: int, cols: int, wall_prob: float = 0.2) -> Grid:
    grid = []
    for _ in range(rows):
        row = []
        for _ in range(cols):
            row.append("#" if random.random() < wall_prob else ".")
        grid.append(row)
    return grid


def pick_free_cell(grid: Grid, used: set) -> Optional[Point]:
    rows, cols = len(grid), len(grid[0])
    candidates = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == "." and (r, c) not in used]
    if not candidates:
        return None
    return random.choice(candidates)


def evaluate_algorithm(name: str, grid: Grid, start: Point, goal: Point, fire_cells: List[Point], dynamic: bool):
    grid_copy = copy.deepcopy(grid)
    s, g = start, goal

    t0 = time.perf_counter()
    if dynamic:
        fn = {"A*": dynamic_astar, "BFS": dynamic_bfs, "GBFS": dynamic_gbfs}[name]
        path, replans = fn(grid_copy, s, g, list(fire_cells))
    else:
        for r, c in fire_cells:
            grid_copy[r][c] = "F"
        fn = {"A*": astar, "BFS": bfs, "GBFS": gbfs}[name]
        path = fn(grid_copy, s, g)
        replans = 0
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    found = path is not None
    steps = (len(path) - 1) if found else 10**9
    return {"algorithm": name, "found": found, "steps": steps, "replans": replans, "runtime_ms": elapsed_ms}


def scenario_features(grid: Grid, start: Point, goal: Point, fire_cells: List[Point]):
    rows, cols = len(grid), len(grid[0])
    wall_count = sum(row.count("#") for row in grid)
    fire_count = len(fire_cells)
    obstacle_density = (wall_count + fire_count) / (rows * cols)
    manhattan = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
    return [rows, cols, wall_count, fire_count, obstacle_density, manhattan]


@dataclass
class ModeRecommender:
    """Simple learned-centroid recommender with a scikit-like predict API."""

    centroids: dict
    default_label: str

    def _distance(self, a: List[float], b: List[float]) -> float:
        return sum((x - y) ** 2 for x, y in zip(a, b))

    def _score_map(self, row: List[float]) -> dict:
        # Convert centroid distances to normalized confidence-like scores.
        # Lower distance => higher score.
        eps = 1e-9
        inv = {}
        for label, centroid in self.centroids.items():
            d = self._distance(row, centroid)
            inv[label] = 1.0 / (d + eps)
        total = sum(inv.values()) or 1.0
        return {label: val / total for label, val in inv.items()}

    def predict(self, x_rows: List[List[float]]) -> List[str]:
        labels = []
        for row in x_rows:
            score_map = self._score_map(row)
            best_label = max(score_map.items(), key=lambda kv: kv[1])[0] if score_map else self.default_label
            labels.append(best_label)
        return labels

    def predict_proba(self, x_rows: List[List[float]]) -> List[dict]:
        return [self._score_map(row) for row in x_rows]


def train_recommender(x_data: List[List[float]], y_labels: List[str]) -> ModeRecommender:
    by_label = {}
    for x, y in zip(x_data, y_labels):
        by_label.setdefault(y, []).append(x)

    centroids = {}
    for label, rows in by_label.items():
        dim = len(rows[0])
        centroid = []
        for i in range(dim):
            centroid.append(sum(r[i] for r in rows) / len(rows))
        centroids[label] = centroid

    default_label = max(by_label.items(), key=lambda kv: len(kv[1]))[0]
    return ModeRecommender(centroids=centroids, default_label=default_label)


def main():
    random.seed(42)
    samples = 900
    x_data = []
    y_labels = []
    raw_results = []
    class_counts = {"A*": 0, "BFS": 0, "GBFS": 0}
    target_per_class = 120

    for _ in range(samples):
        rows = random.randint(5, 10)
        cols = random.randint(5, 10)
        dynamic = random.choice([True, False])
        wall_prob = random.uniform(0.10, 0.30)
        grid = random_grid(rows, cols, wall_prob=wall_prob)

        used = set()
        start = pick_free_cell(grid, used)
        if not start:
            continue
        used.add(start)
        goal = pick_free_cell(grid, used)
        if not goal:
            continue
        used.add(goal)

        fire_cells = []
        fire_count = random.randint(0, max(1, (rows * cols) // 12))
        for _ in range(fire_count):
            f = pick_free_cell(grid, used)
            if f:
                used.add(f)
                fire_cells.append(f)

        outcomes = [
            evaluate_algorithm("A*", grid, start, goal, fire_cells, dynamic),
            evaluate_algorithm("BFS", grid, start, goal, fire_cells, dynamic),
            evaluate_algorithm("GBFS", grid, start, goal, fire_cells, dynamic),
        ]

        valid = [o for o in outcomes if o["found"]]
        if not valid:
            continue

        # Primary objective: path quality (steps + replans).
        # Tie-breaker: runtime to allow GBFS/BFS to win in simple layouts.
        best = min(valid, key=lambda o: (o["steps"], o["replans"], o["runtime_ms"]))

        # Keep training data roughly balanced across labels.
        if class_counts[best["algorithm"]] >= target_per_class:
            continue

        class_counts[best["algorithm"]] += 1
        x_data.append(scenario_features(grid, start, goal, fire_cells))
        y_labels.append(best["algorithm"])
        raw_results.append(
            {
                "rows": rows,
                "cols": cols,
                "dynamic": dynamic,
                "start": start,
                "goal": goal,
                "fires": fire_cells,
                "winner": best["algorithm"],
                "outcomes": outcomes,
            }
        )

        if all(class_counts[k] >= target_per_class for k in class_counts):
            break

    if not x_data:
        raise RuntimeError("No valid training scenarios generated.")

    model = train_recommender(x_data, y_labels)

    out_dir = Path(__file__).resolve().parent
    model_path = out_dir / "mode_model.pkl"
    results_path = out_dir / "comparison_results.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(results_path, "wb") as f:
        pickle.dump(raw_results, f)

    print(f"Saved trained model: {model_path}")
    print(f"Saved comparison dataset: {results_path}")
    print(f"Training samples used: {len(x_data)}")
    print(f"Class distribution: {class_counts}")


if __name__ == "__main__":
    main()
