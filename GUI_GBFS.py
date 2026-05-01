import streamlit as st
import copy
from GBFS import gbfs, dynamic_gbfs

#  PAGE CONFIG
st.set_page_config(
    page_title="Smart Emergency Evacuation Planner — GBFS",
    layout="wide"
)

#  CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@700;800&display=swap');
html, body, [class*="css"] {
    background-color: #080c14; color: #c9d1d9;
    font-family: 'JetBrains Mono', monospace;
}
.title-block {
    background: linear-gradient(120deg, #0d1117 60%, #0d001a);
    border-left: 4px solid #9b59b6; border-radius: 0 10px 10px 0;
    padding: 20px 28px; margin-bottom: 24px;
}
.title-block h1 { font-family:'Syne',sans-serif; font-size:1.9rem; color:#c39bd3; margin:0; letter-spacing:1px; }
.title-block p  { color:#555e6e; font-size:0.8rem; margin:4px 0 0 0; }
.section-label {
    font-family:'Syne',sans-serif; font-size:0.72rem; font-weight:700;
    color:#9b59b6; letter-spacing:2px; text-transform:uppercase; margin-bottom:6px;
}
.grid-wrap {
    display:inline-block; background:#060a0f;
    border:1px solid #1c2333; border-radius:8px; padding:8px; margin:6px 0;
}
table.astar-grid { border-collapse:collapse; }
table.astar-grid td {
    width:42px; height:42px; text-align:center; vertical-align:middle;
    font-size:18px; border:1px solid #0d1117; border-radius:4px;
}
.result-box {
    background:#060a0f; border:1px solid #1c2333; border-left:3px solid #9b59b6;
    border-radius:6px; padding:10px 14px; font-size:0.82rem; color:#8b949e;
    margin-top:8px; word-break:break-all; line-height:1.6;
}
.result-box.success { border-left-color:#3fb950; color:#3fb950; }
.result-box.failure { border-left-color:#f85149; color:#f85149; }
.result-box.replan  { border-left-color:#0070f3; color:#79c0ff; }
.stat-row { display:flex; gap:12px; margin-top:10px; flex-wrap:wrap; }
.stat-chip {
    background:#0d1117; border:1px solid #1c2333; border-radius:8px;
    padding:8px 16px; text-align:center; min-width:90px;
}
.stat-chip .val { font-family:'Syne',sans-serif; font-size:1.3rem; color:#c39bd3; }
.stat-chip .lbl { font-size:0.68rem; color:#555e6e; text-transform:uppercase; letter-spacing:1px; }
.legend-row { display:flex; gap:10px; flex-wrap:wrap; margin:10px 0 18px 0; }
.legend-chip {
    display:flex; align-items:center; gap:6px; background:#0d1117;
    border:1px solid #1c2333; border-radius:20px; padding:4px 12px;
    font-size:0.74rem; color:#8b949e;
}
div.stButton > button {
    font-family:'JetBrains Mono',monospace; background:#9b59b6; color:white;
    border:none; border-radius:6px; padding:7px 18px; font-size:0.85rem;
    width:100%; transition:background 0.2s;
}
div.stButton > button:hover { background:#c39bd3; color:#080c14; }
div[data-testid="stSelectbox"] label,
div[data-testid="stNumberInput"] label,
div[data-testid="stRadio"] label { color:#8b949e !important; font-size:0.8rem; }
section[data-testid="stSidebar"] { background:#0a0d16; border-right:1px solid #1a0033; }
.fire-tag {
    display:inline-block; background:#2b0a00; border:1px solid #ff4500;
    border-radius:4px; padding:2px 8px; font-size:0.75rem; color:#ff6644; margin:2px;
}
.info-box {
    background:#0d1117; border:1px solid #1c2333; border-radius:8px;
    padding:12px 16px; font-size:0.8rem; color:#555e6e; line-height:1.7; margin-top:8px;
}
</style>
""", unsafe_allow_html=True)

#  Preset test cases
PRESETS = {
    "TC1 — Normal GBFS": {
        "rows": 3, "cols": 4,
        "grid": [['.','.','.','.'],['#','#','.','#'],['.','.','.','.']],
        "start": (0,0), "goal": (2,3), "fire": [], "mode": "gbfs",
    },
    "TC2 — Fire Avoidance": {
        "rows": 3, "cols": 4,
        "grid": [['.','.','.','.'],['.','#','.','.'],['.','.','.','.']],
        "start": (0,0), "goal": (2,3), "fire": [(0,2),(1,2)], "mode": "gbfs",
    },
    "TC3A — Dynamic Replan (fire @1,0)": {
        "rows": 3, "cols": 4,
        "grid": [['.','.','.','.'],['.','.','.','.'],['.','.','.','.']],
        "start": (0,0), "goal": (2,3), "fire": [(1,0)], "mode": "dynamic",
    },
    "TC3B — Dynamic Replan (fire @1,1)": {
        "rows": 3, "cols": 4,
        "grid": [['.','.','.','.'],['.','.','.','.'],['.','.','.','.']],
        "start": (0,0), "goal": (2,3), "fire": [(1,1)], "mode": "dynamic",
    },
    "TC4 — No Possible Path": {
        "rows": 3, "cols": 4,
        "grid": [['.','#','#','#'],['#','.','.','.'],['#','.','.','.']],
        "start": (0,0), "goal": (2,3), "fire": [], "mode": "gbfs",
    },
    "TC5 — GBFS Suboptimal (12 steps vs A* 10)": {
        "rows": 5, "cols": 7,
        "grid": [
            ['.', '.', '#', '.', '#', '#', '.'],
            ['.', '#', '#', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '#', '.', '.', '.'],
            ['.', '#', '.', '#', '.', '#', '.'],
        ],
        "start": (0,0), "goal": (4,6), "fire": [], "mode": "gbfs",
    },
    "Custom (blank grid)": {
        "rows": 6, "cols": 8,
        "grid": None, "start": None, "goal": None, "fire": [], "mode": "gbfs",
    },
}

def init_state(rows, cols):
    st.session_state.grid       = [['.' for _ in range(cols)] for _ in range(rows)]
    st.session_state.start      = None
    st.session_state.goal       = None
    st.session_state.fire_cells = []
    st.session_state.result     = None
    st.session_state.rows       = rows
    st.session_state.cols       = cols

def load_preset(name):
    p = PRESETS[name]
    rows, cols = p["rows"], p["cols"]
    st.session_state.rows       = rows
    st.session_state.cols       = cols
    st.session_state.grid       = copy.deepcopy(p["grid"]) if p["grid"] else [['.']*cols for _ in range(rows)]
    st.session_state.start      = p["start"]
    st.session_state.goal       = p["goal"]
    st.session_state.fire_cells = list(p["fire"])
    for r, c in st.session_state.fire_cells:
        st.session_state.grid[r][c] = 'F'
    st.session_state.result     = None
    st.session_state.algo_mode  = p["mode"]

if 'grid' not in st.session_state:
    init_state(5, 7)
if 'algo_mode' not in st.session_state:
    st.session_state.algo_mode = 'gbfs'

#  HELPERS
def set_cell(r, c, value):
    g = st.session_state.grid
    if (r, c) == st.session_state.start and value != 'S':
        st.session_state.start = None
    if (r, c) == st.session_state.goal and value != 'E':
        st.session_state.goal = None
    if (r, c) in st.session_state.fire_cells and value != 'F':
        st.session_state.fire_cells.remove((r, c))

    if value == 'S':
        if st.session_state.start:
            pr, pc = st.session_state.start
            g[pr][pc] = '.'
        st.session_state.start = (r, c)
        g[r][c] = '.'
    elif value == 'E':
        if st.session_state.goal:
            pr, pc = st.session_state.goal
            g[pr][pc] = '.'
        st.session_state.goal = (r, c)
        g[r][c] = '.'
    elif value == 'F':
        g[r][c] = 'F'
        if (r, c) not in st.session_state.fire_cells:
            st.session_state.fire_cells.append((r, c))
    else:
        g[r][c] = value
    st.session_state.result = None


CELL_STYLE = {
    'S': ('🟩', '#0d2b0d'),
    'E': ('🏁', '#0d1a2b'),
    '#': ('🟫', '#1e1208'),
    'F': ('🔥', '#2b1000'),
    'P': ('🟣', '#1a0d2b'),
    '.': ('⬛', '#0a0e15'),
}

def render_grid_html(grid, path=None):
    path_set = set(path) if path else set()
    start    = st.session_state.start
    goal     = st.session_state.goal
    html = '<div class="grid-wrap"><table class="astar-grid">'
    for r, row in enumerate(grid):
        html += '<tr>'
        for c, cell in enumerate(row):
            pos = (r, c)
            if pos == start:
                emoji, bg = CELL_STYLE['S']
            elif pos == goal:
                emoji, bg = CELL_STYLE['E']
            elif cell == '#':
                emoji, bg = CELL_STYLE['#']
            elif cell == 'F':
                emoji, bg = CELL_STYLE['F']
            elif pos in path_set and pos not in (start, goal):
                emoji, bg = CELL_STYLE['P']
            else:
                emoji, bg = CELL_STYLE['.']
            html += f'<td style="background:{bg};">{emoji}</td>'
        html += '</tr>'
    html += '</table></div>'
    return html


def result_html(res):
    if res is None:
        return ''
    path = res['path']
    mode = res['mode']
    if path is None:
        return '<div class="result-box failure">❌ No path found — destination is unreachable.</div>'
    steps        = len(path) - 1
    coords       = ' → '.join(str(p) for p in path)
    css          = 'replan' if mode == 'dynamic' else 'success'
    icon         = '🔄' if mode == 'dynamic' else '✅'
    replan_badge = f'&nbsp;·&nbsp; 🔥 {res["replannings"]} replan(s)' if mode == 'dynamic' else ''
    return (f'<div class="result-box {css}">'
            f'{icon} Path found &nbsp;·&nbsp; <b>{steps} steps</b>{replan_badge}'
            f'<br><span style="color:#333e4e;font-size:0.72rem;">{coords}</span>'
            f'</div>')


#  TITLE
st.markdown("""
<div class="title-block">
  <h1>🏢 Smart Emergency Evacuation Planner</h1>
  <p>Greedy Best-First Search · Manhattan Heuristic · Fire Spread Simulation · Fully Interactive Grid</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="legend-row">
  <div class="legend-chip">🟩 Start (S)</div>
  <div class="legend-chip">🏁 Goal (E)</div>
  <div class="legend-chip">🟫 Wall (#)</div>
  <div class="legend-chip">🔥 Fire (F)</div>
  <div class="legend-chip">🟣 Path</div>
  <div class="legend-chip">⬛ Empty (.)</div>
</div>
""", unsafe_allow_html=True)

#  SIDEBAR
with st.sidebar:
    st.markdown('<div class="section-label"> Preset Test Cases</div>', unsafe_allow_html=True)
    preset_choice = st.selectbox("Preset", list(PRESETS.keys()), label_visibility="collapsed")

    if st.button("⬇  Load Preset"):
        load_preset(preset_choice)
        st.rerun()

    st.markdown("---")
    st.markdown('<div class="section-label"> Grid Size</div>', unsafe_allow_html=True)
    new_rows = st.number_input("Rows", min_value=2, max_value=12,
                               value=st.session_state.rows, step=1)
    new_cols = st.number_input("Cols", min_value=2, max_value=12,
                               value=st.session_state.cols, step=1)
    if st.button("🔄 Resize (clears grid)"):
        init_state(int(new_rows), int(new_cols))
        st.rerun()

    st.markdown("---")
    st.markdown('<div class="section-label"> Algorithm Mode</div>', unsafe_allow_html=True)

    algo_options = ["GBFS (static fire = obstacle)", "Dynamic GBFS (fire triggers replan)"]

    st.session_state["algo_radio"] = (
        algo_options[1] if st.session_state.algo_mode == "dynamic" else algo_options[0]
    )

    algo = st.radio(
        "Algorithm",
        algo_options,
        key="algo_radio",
        label_visibility="collapsed"
    )
    st.session_state.algo_mode = "dynamic" if "Dynamic" in st.session_state["algo_radio"] else "gbfs"

#  MAIN LAYOUT
left_col, right_col = st.columns([3, 2], gap="large")

with left_col:
    st.markdown('<div class="section-label"> Current Grid</div>', unsafe_allow_html=True)

    path_to_show = st.session_state.result['path'] if st.session_state.result else None
    st.markdown(render_grid_html(st.session_state.grid, path=path_to_show),
                unsafe_allow_html=True)

    st.markdown("")

    # RUN button
    if st.button("▶  RUN PATHFINDER"):
        s = st.session_state.start
        g = st.session_state.goal

        if s is None or g is None:
            st.error("⚠️ Place both **Start (S)** and **Goal (E)** before running.")
        else:
            grid_copy = copy.deepcopy(st.session_state.grid)
            fire_copy = list(st.session_state.fire_cells)
            sr, sc    = s
            gr, gc    = g
            grid_copy[sr][sc] = '.'
            grid_copy[gr][gc] = '.'

            use_dynamic = (st.session_state.algo_mode == 'dynamic')

            if use_dynamic:
                for r, c in fire_copy:
                    grid_copy[r][c] = '.'

                path, replannings = dynamic_gbfs(grid_copy, s, g, fire_copy)

                st.session_state.result = {
                    'path': path, 'mode': 'dynamic', 'replannings': replannings
                }
            else:
                # Mark fire as static obstacles for regular GBFS
                for r, c in fire_copy:
                    grid_copy[r][c] = 'F'
                path = gbfs(grid_copy, s, g)
                st.session_state.result = {
                    'path': path, 'mode': 'gbfs', 'replannings': 0
                }
            st.rerun()

# Right panel
with right_col:
    st.markdown('<div class="section-label"> Result</div>', unsafe_allow_html=True)

    res = st.session_state.result
    if res:
        st.markdown(result_html(res), unsafe_allow_html=True)
        path = res['path']
        st.markdown(f"""
        <div class="stat-row">
          <div class="stat-chip">
            <div class="val">{len(path)-1 if path else '—'}</div>
            <div class="lbl">Steps</div>
          </div>
          <div class="stat-chip">
            <div class="val">{res['replannings']}</div>
            <div class="lbl">Replannings</div>
          </div>
          <div class="stat-chip">
            <div class="val">{len(st.session_state.fire_cells)}</div>
            <div class="lbl">Fire Cells</div>
          </div>
          <div class="stat-chip">
            <div class="val">{st.session_state.rows}×{st.session_state.cols}</div>
            <div class="lbl">Grid Size</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        if path:
            with st.expander(" Full path coordinates"):
                st.code('\n'.join(f"Step {i:>2}: {p}" for i, p in enumerate(path)),
                        language='text')
    else:
        st.markdown("""
        <div class="result-box" style="color:#333e4e;">
           Configure your grid and press <b>▶ RUN PATHFINDER</b>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-label"> Grid State</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-box">
      Start &nbsp; → &nbsp;<b style="color:#3fb950;">{st.session_state.start}</b><br>
      Goal &nbsp;&nbsp; → &nbsp;<b style="color:#79c0ff;">{st.session_state.goal}</b><br>
      Walls &nbsp; → &nbsp;{sum(row.count('#') for row in st.session_state.grid)} cells<br>
      Fire &nbsp;&nbsp; → &nbsp;<b style="color:#ff4500;">
        {st.session_state.fire_cells if st.session_state.fire_cells else 'none'}
      </b>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="section-label"> Paint a Cell</div>', unsafe_allow_html=True)
    draw_mode = st.radio(
        "Cell type",
        ["Wall (#)", "Fire (F)", "Start (S)", "Goal (E)", "Erase (.)"],
        label_visibility="collapsed"
    )
    mode_map = {"Wall (#)":'#', "Fire (F)":'F', "Start (S)":'S',
                "Goal (E)":'E', "Erase (.)":'.'}

    pc1, pc2 = st.columns(2)
    with pc1:
        paint_r = st.number_input("Row", min_value=0,
                                  max_value=st.session_state.rows - 1,
                                  value=0, step=1, key="pr")
    with pc2:
        paint_c = st.number_input("Col", min_value=0,
                                  max_value=st.session_state.cols - 1,
                                  value=0, step=1, key="pc")

    if st.button("  Apply Paint"):
        set_cell(int(paint_r), int(paint_c), mode_map[draw_mode])
        st.rerun()

    if st.button("  Clear All"):
        init_state(st.session_state.rows, st.session_state.cols)
        st.rerun()