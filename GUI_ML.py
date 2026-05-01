import copy
import pickle
from pathlib import Path

import streamlit as st

from Astar import astar, dynamic_astar
from BFS import bfs, dynamic_bfs
from GBFS import gbfs, dynamic_gbfs


st.set_page_config(page_title="Smart Evacuation Planner - ML", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@700;800&display=swap');
html, body, [class*="css"] {
    background-color: #080c14; color: #c9d1d9;
    font-family: 'JetBrains Mono', monospace;
}
.title-block {
    background: linear-gradient(120deg, #0d1117 60%, #1a0d00);
    border-left: 4px solid #ff4500; border-radius: 0 10px 10px 0;
    padding: 20px 28px; margin-bottom: 24px;
}
.title-block h1 { font-family:'Syne',sans-serif; font-size:1.9rem; color:#ff6b35; margin:0; letter-spacing:1px; }
.title-block p  { color:#555e6e; font-size:0.8rem; margin:4px 0 0 0; }
.section-label {
    font-family:'Syne',sans-serif; font-size:0.72rem; font-weight:700;
    color:#ff4500; letter-spacing:2px; text-transform:uppercase; margin-bottom:6px;
}
.grid-wrap {
    display:inline-block; background:#060a0f;
    border:1px solid #1c2333; border-radius:8px; padding:8px; margin:6px 0;
}
table.astar-grid { border-collapse:collapse; }
table.astar-grid td {
    width:42px; height:42px; text-align:center; vertical-align:middle;
    font-size:16px; border:1px solid #0d1117; border-radius:4px;
}
.result-box {
    background:#060a0f; border:1px solid #1c2333; border-left:3px solid #ff4500;
    border-radius:6px; padding:10px 14px; font-size:0.82rem; color:#8b949e;
    margin-top:8px; line-height:1.6;
}
.result-box.success { border-left-color:#3fb950; color:#3fb950; }
.legend-row { display:flex; gap:10px; flex-wrap:wrap; margin:10px 0 18px 0; }
.legend-chip {
    display:flex; align-items:center; gap:6px; background:#0d1117;
    border:1px solid #1c2333; border-radius:20px; padding:4px 12px;
    font-size:0.74rem; color:#8b949e;
}
.risk-low { color:#3fb950; }
.risk-mid { color:#f2cc60; }
.risk-high { color:#f85149; }
div.stButton > button {
    font-family:'JetBrains Mono',monospace; background:#ff4500; color:white;
    border:none; border-radius:6px; padding:7px 18px; font-size:0.85rem;
    width:100%; transition:background 0.2s;
}
div.stButton > button:hover { background:#ff6644; color:white; }
section[data-testid="stSidebar"] { background:#0a0d16; border-right:1px solid #1a1a33; }
</style>
""",
    unsafe_allow_html=True,
)

PRESETS = {
    "TC1 — Normal": {
        "rows": 3,
        "cols": 4,
        "grid": [[".", ".", ".", "."], ["#", "#", ".", "#"], [".", ".", ".", "."]],
        "start": (0, 0),
        "goal": (2, 3),
        "fire": [],
        "mode": "astar",
    },
    "TC2 — Fire Avoidance": {
        "rows": 5,
        "cols": 6,
        "grid": [
            [".", ".", ".", ".", ".", "."],
            [".", "#", "#", ".", "#", "."],
            [".", ".", ".", ".", "#", "."],
            [".", "#", ".", ".", ".", "."],
            [".", ".", ".", "#", ".", "."],
        ],
        "start": (0, 0),
        "goal": (4, 5),
        "fire": [(1, 3), (3, 2)],
        "mode": "astar",
    },
    "TC- Small maze (BFS often best by time)": {
        "rows": 6,
        "cols": 6,
        "grid": [
            [".", ".", "#", ".", ".", "."],
            ["#", ".", "#", ".", "#", "."],
            [".", ".", ".", ".", "#", "."],
            [".", "#", "#", ".", ".", "."],
            [".", ".", ".", "#", ".", "#"],
            [".", "#", ".", ".", ".", "."],
        ],
        "start": (0, 0),
        "goal": (5, 5),
        "fire": [],
        "mode": "astar",
    },
    "TC- Dense obstacles + fire": {
        "rows": 7,
        "cols": 7,
        "grid": [
            [".", ".", ".", "#", ".", ".", "."],
            [".", "#", ".", "#", ".", "#", "."],
            [".", "#", ".", ".", ".", "#", "."],
            [".", ".", ".", "#", ".", ".", "."],
            ["#", "#", ".", "#", ".", "#", "."],
            [".", ".", ".", ".", ".", "#", "."],
            [".", "#", "#", "#", ".", ".", "."],
        ],
        "start": (0, 0),
        "goal": (6, 6),
        "fire": [(1, 4), (3, 1), (5, 5)],
        "mode": "dynamic",
    },
    "TC- Open field (GBFS often best by time)": {
        "rows": 10,
        "cols": 10,
        "grid": [
            [".", ".", ".", ".", ".", ".", ".", ".", ".", "."],
            [".", ".", ".", ".", ".", ".", ".", ".", ".", "."],
            [".", ".", ".", ".", ".", ".", ".", ".", ".", "."],
            [".", ".", ".", ".", "#", ".", ".", ".", ".", "."],
            [".", ".", ".", ".", "#", ".", ".", ".", ".", "."],
            [".", ".", ".", ".", ".", ".", ".", ".", ".", "."],
            [".", ".", ".", ".", ".", ".", ".", ".", ".", "."],
            [".", ".", ".", ".", ".", ".", ".", ".", ".", "."],
            [".", ".", ".", ".", ".", ".", ".", ".", ".", "."],
            [".", ".", ".", ".", ".", ".", ".", ".", ".", "."],
        ],
        "start": (0, 0),
        "goal": (9, 9),
        "fire": [],
        "mode": "astar",
    },
    "Custom (blank grid)": {
        "rows": 6,
        "cols": 8,
        "grid": None,
        "start": None,
        "goal": None,
        "fire": [],
        "mode": "astar",
    },
}

CELL_STYLE = {
    "S": ("🟩", "#0d2b0d"),
    "E": ("🏁", "#0d1a2b"),
    "#": ("🟫", "#1e1208"),
    "F": ("🔥", "#2b1000"),
    "P": ("🔵", "#0d1a33"),
    ".": ("⬛", "#0a0e15"),
}


def init_state(rows, cols):
    st.session_state.grid = [["." for _ in range(cols)] for _ in range(rows)]
    st.session_state.start = None
    st.session_state.goal = None
    st.session_state.fire_cells = []
    st.session_state.result = None
    st.session_state.compare_result = None
    st.session_state.rows = rows
    st.session_state.cols = cols
    st.session_state.algo_mode = "astar"


def load_preset(name):
    p = PRESETS[name]
    rows, cols = p["rows"], p["cols"]
    st.session_state.rows = rows
    st.session_state.cols = cols
    st.session_state.grid = copy.deepcopy(p["grid"]) if p["grid"] else [["."] * cols for _ in range(rows)]
    st.session_state.start = p["start"]
    st.session_state.goal = p["goal"]
    st.session_state.fire_cells = list(p["fire"])
    for r, c in st.session_state.fire_cells:
        st.session_state.grid[r][c] = "F"
    st.session_state.result = None
    st.session_state.compare_result = None
    st.session_state.algo_mode = p["mode"]


def set_cell(r, c, token):
    grid = st.session_state.grid
    prev = grid[r][c]
    if prev == "F" and (r, c) in st.session_state.fire_cells:
        st.session_state.fire_cells.remove((r, c))
    if st.session_state.start == (r, c):
        st.session_state.start = None
    if st.session_state.goal == (r, c):
        st.session_state.goal = None

    if token == "S":
        st.session_state.start = (r, c)
        grid[r][c] = "."
    elif token == "E":
        st.session_state.goal = (r, c)
        grid[r][c] = "."
    elif token == "F":
        if (r, c) != st.session_state.start and (r, c) != st.session_state.goal:
            grid[r][c] = "F"
            if (r, c) not in st.session_state.fire_cells:
                st.session_state.fire_cells.append((r, c))
    elif token == "#":
        if (r, c) != st.session_state.start and (r, c) != st.session_state.goal:
            grid[r][c] = "#"
    else:
        grid[r][c] = "."


def extract_features(grid, fire_cells, start, goal):
    rows, cols = len(grid), len(grid[0])
    wall_count = sum(row.count("#") for row in grid)
    fire_count = len(fire_cells)
    obstacle_density = (wall_count + fire_count) / (rows * cols)
    manhattan = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
    return [rows, cols, wall_count, fire_count, obstacle_density, manhattan]


def load_mode_model():
    model_path = Path(__file__).with_name("mode_model.pkl")
    if not model_path.exists():
        return None, f"{model_path.name} not found. Run train_mode.py first."

    class ModeModelUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "__main__" and name == "ModeRecommender":
                from train_mode import ModeRecommender

                return ModeRecommender
            return super().find_class(module, name)

    try:
        with open(model_path, "rb") as f:
            return ModeModelUnpickler(f).load(), None
    except Exception as exc:
        return None, f"Could not load {model_path.name}: {exc}"


def load_fire_model():
    model_path = Path(__file__).with_name("fire_model.pkl")
    if not model_path.exists():
        return None, f"{model_path.name} not found. Run train_mode.py first."

    class FireModelUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "__main__" and name == "FireSpreadModel":
                from train_mode import FireSpreadModel

                return FireSpreadModel
            return super().find_class(module, name)

    try:
        with open(model_path, "rb") as f:
            return FireModelUnpickler(f).load(), None
    except Exception as exc:
        return None, f"Could not load {model_path.name}: {exc}"


def fire_cell_features(grid, fire_cells, pos):
    rows, cols = len(grid), len(grid[0])
    wall_count = sum(row.count("#") for row in grid)
    fire_count = len(fire_cells)
    obstacle_density = (wall_count + fire_count) / (rows * cols)
    r, c = pos

    burning_neighbors = 0
    fire_set = set(fire_cells)
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) in fire_set:
            burning_neighbors += 1

    if fire_cells:
        min_dist = min(abs(r - fr) + abs(c - fc) for fr, fc in fire_cells)
    else:
        min_dist = rows + cols

    return [rows, cols, wall_count, fire_count, obstacle_density, burning_neighbors, min_dist]


def run_algorithm(grid, s, g, fire_cells, algo_name, dynamic):
    import time
    grid_copy = copy.deepcopy(grid)
    t0 = time.perf_counter()
    if dynamic:
        fn = {"A*": dynamic_astar, "BFS": dynamic_bfs, "GBFS": dynamic_gbfs}[algo_name]
        for r, c in fire_cells:
            grid_copy[r][c] = "."
        path, replans = fn(grid_copy, s, g, list(fire_cells))
    else:
        fn = {"A*": astar, "BFS": bfs, "GBFS": gbfs}[algo_name]
        for r, c in fire_cells:
            grid_copy[r][c] = "F"
        path = fn(grid_copy, s, g)
        replans = 0
    runtime_ms = (time.perf_counter() - t0) * 1000.0
    return path, replans, runtime_ms


def predict_next_fire_cells(grid, fire_cells, fire_model, top_k=3):
    """Predict top-k most likely next-fire cells using trained fire model."""
    if not fire_cells or fire_model is None or top_k <= 0:
        return []
    rows, cols = len(grid), len(grid[0])
    candidates = []
    feature_rows = []
    fire_set = set(fire_cells)
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] in ("#", "F") or (r, c) in fire_set:
                continue
            candidates.append((r, c))
            feature_rows.append(fire_cell_features(grid, fire_cells, (r, c)))
    if not candidates:
        return []
    probs = fire_model.predict_proba(feature_rows)
    ranked = sorted(zip(candidates, probs), key=lambda item: item[1], reverse=True)
    return [(cell, float(prob)) for cell, prob in ranked[: min(top_k, len(ranked))]]


def run_algorithm_with_predicted_fire(grid, s, g, fire_cells, algo_name, dynamic, predicted_fire):
    """Run algorithm after reserving only top predicted next-fire cell as blocked."""
    grid_copy = copy.deepcopy(grid)
    if predicted_fire is not None:
        pr, pc = predicted_fire
        if (pr, pc) not in (s, g) and grid_copy[pr][pc] not in ("#", "F"):
            grid_copy[pr][pc] = "F"
    return run_algorithm(grid_copy, s, g, fire_cells, algo_name, dynamic)


def compare_algorithms_live(grid, s, g, fire_cells, dynamic, fire_model, fire_top_k):
    rows_out = []
    predicted_fire_rows = predict_next_fire_cells(grid, fire_cells, fire_model, top_k=fire_top_k)
    top_predicted_fire = predicted_fire_rows[0][0] if predicted_fire_rows else None
    for algo_name in ["A*", "BFS", "GBFS"]:
        path, replans, runtime_ms = run_algorithm_with_predicted_fire(
            grid, s, g, fire_cells, algo_name, dynamic, top_predicted_fire
        )
        rows_out.append(
            {
                "Algorithm": algo_name,
                "Found Path": "Yes" if path else "No",
                "Steps": (len(path) - 1) if path else None,
                "Replans": replans,
                "Runtime (ms)": round(runtime_ms, 3),
            }
        )
    valid = [r for r in rows_out if r["Found Path"] == "Yes"]
    if valid:
        best = min(
            valid,
            key=lambda r: (
                r["Steps"] if r["Steps"] is not None else 10**9,
                r["Replans"],
                r["Runtime (ms)"],
            ),
        )["Algorithm"]
    else:
        best = None
    for r in rows_out:
        r["Best"] = "✅" if best and r["Algorithm"] == best else ""
    return rows_out, best, predicted_fire_rows


def risk_probabilities(grid, fire_cells, fire_model):
    rows, cols = len(grid), len(grid[0])
    probs = [[0.0 for _ in range(cols)] for _ in range(rows)]
    if not fire_cells or fire_model is None:
        return probs
    feature_rows = []
    refs = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == "#":
                probs[r][c] = 0.0
                continue
            if (r, c) in fire_cells:
                probs[r][c] = 1.0
                continue
            refs.append((r, c))
            feature_rows.append(fire_cell_features(grid, fire_cells, (r, c)))
    pred_probs = fire_model.predict_proba(feature_rows) if feature_rows else []
    for (r, c), p in zip(refs, pred_probs):
        probs[r][c] = max(0.0, min(1.0, float(p)))
    return probs


def color_for_probability(p):
    red = int(255 * p)
    green = int(180 * (1.0 - p))
    blue = 40
    return f"rgb({red},{green},{blue})"


def render_grid_html(grid, path=None, risk=None):
    path_set = set(path) if path else set()
    start = st.session_state.start
    goal = st.session_state.goal
    html = '<div class="grid-wrap"><table class="astar-grid">'
    for r, row in enumerate(grid):
        html += "<tr>"
        for c, cell in enumerate(row):
            pos = (r, c)
            if pos == start:
                emoji, bg = CELL_STYLE["S"]
            elif pos == goal:
                emoji, bg = CELL_STYLE["E"]
            elif cell == "#":
                emoji, bg = CELL_STYLE["#"]
            elif cell == "F":
                emoji, bg = CELL_STYLE["F"]
            elif pos in path_set and pos not in (start, goal):
                emoji, bg = CELL_STYLE["P"]
            else:
                emoji, bg = CELL_STYLE["."]

            if risk is not None and cell != "#":
                rp = risk[r][c]
                bg = color_for_probability(rp)
                emoji = f"{int(rp * 100)}%"
            html += f'<td style="background:{bg};">{emoji}</td>'
        html += "</tr>"
    html += "</table></div>"
    return html


if "grid" not in st.session_state:
    init_state(6, 8)

st.markdown(
    """
<div class="title-block">
  <h1>🏢 Smart Emergency Evacuation Planner (ML)</h1>
  <p>Pathfinding + Learned Algorithm Recommendation + Fire Risk Probability View</p>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="legend-row">
  <div class="legend-chip">🟩 Start</div>
  <div class="legend-chip">🏁 Goal</div>
  <div class="legend-chip">🟫 Wall</div>
  <div class="legend-chip">🔥 Fire</div>
  <div class="legend-chip">🔵 Path</div>
  <div class="legend-chip">Risk map = % per cell</div>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown('<div class="section-label">Preset</div>', unsafe_allow_html=True)
    preset_choice = st.selectbox("Preset", list(PRESETS.keys()), label_visibility="collapsed")
    if st.button("⬇ Load Preset"):
        load_preset(preset_choice)
        st.rerun()

    st.markdown("---")
    st.markdown('<div class="section-label">Grid Size</div>', unsafe_allow_html=True)
    new_rows = st.number_input("Rows", min_value=2, max_value=12, value=st.session_state.rows)
    new_cols = st.number_input("Cols", min_value=2, max_value=12, value=st.session_state.cols)
    if st.button("🔄 Resize (clear)"):
        init_state(int(new_rows), int(new_cols))
        st.rerun()

    st.markdown("---")
    st.markdown('<div class="section-label">Mode</div>', unsafe_allow_html=True)
    mode = st.radio("Mode", ["A* static", "Dynamic"], label_visibility="collapsed")
    st.session_state.algo_mode = "dynamic" if mode == "Dynamic" else "astar"
    fire_top_k = 3

left, right = st.columns([3, 2], gap="large")

with left:
    st.markdown('<div class="section-label">Grid</div>', unsafe_allow_html=True)
    path_to_show = st.session_state.result["path"] if st.session_state.result else None
    st.markdown(render_grid_html(st.session_state.grid, path=path_to_show), unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    fire_model, fire_model_error = load_fire_model()
    with c1:
        if st.button("▶ Run Recommended"):
            s, g = st.session_state.start, st.session_state.goal
            if s is None or g is None:
                st.error("Place Start and Goal first.")
            else:
                model, model_error = load_mode_model()
                if model is None:
                    st.error(model_error)
                else:
                    feats = extract_features(st.session_state.grid, st.session_state.fire_cells, s, g)
                    algo_name = model.predict([feats])[0]
                    # Validate recommendation against current grid's measured performance.
                    _, live_best, predicted_fire = compare_algorithms_live(
                        st.session_state.grid,
                        s,
                        g,
                        st.session_state.fire_cells,
                        st.session_state.algo_mode == "dynamic",
                        fire_model,
                        fire_top_k,
                    )
                    algo_name = live_best or algo_name
                    predicted_cells = predicted_fire[0][0] if predicted_fire else None
                    path, replans = run_algorithm_with_predicted_fire(
                        st.session_state.grid,
                        s,
                        g,
                        st.session_state.fire_cells,
                        algo_name,
                        st.session_state.algo_mode == "dynamic",
                        predicted_cells,
                    )[:2]
                    st.session_state.result = {"path": path, "mode": algo_name, "replannings": replans}
                    st.rerun()
    with c2:
        if st.button("📊 Compare All"):
            s, g = st.session_state.start, st.session_state.goal
            if s is None or g is None:
                st.error("Place Start and Goal first.")
            else:
                rows_out, _, _ = compare_algorithms_live(
                    st.session_state.grid,
                    s,
                    g,
                    st.session_state.fire_cells,
                    st.session_state.algo_mode == "dynamic",
                    fire_model,
                    fire_top_k,
                )
                st.session_state.compare_result = rows_out

with right:
    st.markdown('<div class="section-label">ML Recommendation</div>', unsafe_allow_html=True)
    model, model_error = load_mode_model()
    s, g = st.session_state.start, st.session_state.goal
    if model and s and g:
        live_rows, live_best, predicted_fire = compare_algorithms_live(
            st.session_state.grid,
            s,
            g,
            st.session_state.fire_cells,
            st.session_state.algo_mode == "dynamic",
            fire_model,
            fire_top_k,
        )
        any_path = any(r["Found Path"] == "Yes" for r in live_rows)

        feats = extract_features(st.session_state.grid, st.session_state.fire_cells, s, g)
        raw_probs = model.predict_proba([feats])[0]
        probs = {label: float(raw_probs.get(label, 0.0)) for label in ["A*", "BFS", "GBFS"]}
        if not any_path:
            pred = "None"
            probs = {"A*": 0.0, "BFS": 0.0, "GBFS": 0.0}
            st.markdown(
                '<div class="result-box">❌ No path available for current setup. Recommendation disabled.</div>',
                unsafe_allow_html=True,
            )
        else:
            pred = model.predict([feats])[0]
            st.markdown(
                f'<div class="result-box success">🤖 Recommended: <b>{pred}</b></div>',
                unsafe_allow_html=True,
            )
        if live_best and pred != "None" and live_best != pred:
            st.markdown(
                f'<div class="result-box">⚖️ Live benchmark winner on current grid: <b>{live_best}</b></div>',
                unsafe_allow_html=True,
            )
        if predicted_fire:
            cells_text = ", ".join([f"{cell} ({prob:.1%})" for cell, prob in predicted_fire])
            st.markdown(
                f'<div class="result-box">🔥 Predicted next-fire cells (top {len(predicted_fire)}): <b>{cells_text}</b> (only top-1 is blocked for planning)</div>',
                unsafe_allow_html=True,
            )
        elif fire_model is None:
            st.markdown(f'<div class="result-box">{fire_model_error}</div>', unsafe_allow_html=True)
        for label in ["A*", "BFS", "GBFS"]:
            p = float(probs.get(label, 0.0))
            st.write(f"{label}: {p:.2%}")
            st.progress(int(p * 100))
    else:
        st.markdown(
            f'<div class="result-box">{model_error or "Load model + place Start/Goal to see ML probability."}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown('<div class="section-label">Fire Risk Probability Map</div>', unsafe_allow_html=True)
    rp = risk_probabilities(st.session_state.grid, st.session_state.fire_cells, fire_model)
    st.markdown(render_grid_html(st.session_state.grid, risk=rp), unsafe_allow_html=True)
    if st.session_state.fire_cells:
        flat = [p for row in rp for p in row]
        avg_risk = sum(flat) / len(flat)
        level = "LOW"
        css = "risk-low"
        if avg_risk >= 0.66:
            level = "HIGH"
            css = "risk-high"
        elif avg_risk >= 0.33:
            level = "MEDIUM"
            css = "risk-mid"
        st.markdown(f"Overall Risk: <span class='{css}'><b>{level}</b></span> ({avg_risk:.1%})", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-label">Paint Cell</div>', unsafe_allow_html=True)
    draw_mode = st.radio(
        "Cell type",
        ["Wall (#)", "Fire (F)", "Start (S)", "Goal (E)", "Erase (.)"],
        label_visibility="collapsed",
    )
    mode_map = {"Wall (#)": "#", "Fire (F)": "F", "Start (S)": "S", "Goal (E)": "E", "Erase (.)": "."}
    rc1, rc2 = st.columns(2)
    with rc1:
        r = st.number_input("Row", min_value=0, max_value=st.session_state.rows - 1, value=0, key="ml_r")
    with rc2:
        c = st.number_input("Col", min_value=0, max_value=st.session_state.cols - 1, value=0, key="ml_c")
    if st.button("Apply Paint"):
        set_cell(int(r), int(c), mode_map[draw_mode])
        st.rerun()
    if st.button("Clear All"):
        init_state(st.session_state.rows, st.session_state.cols)
        st.rerun()

    if st.session_state.result:
        st.markdown("---")
        result = st.session_state.result
        path = result["path"]
        if path:
            st.write(f"Mode used: {result['mode']}")
            st.write(f"Steps: {len(path)-1}")
            st.write(f"Replans: {result['replannings']}")
        else:
            st.write("No path found.")

    if st.session_state.compare_result:
        st.markdown("---")
        st.markdown('<div class="section-label">Comparison Table</div>', unsafe_allow_html=True)
        st.table(st.session_state.compare_result)
