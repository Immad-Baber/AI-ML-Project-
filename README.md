# Smart Emergency Evacuation Planner

An interactive pathfinding and emergency evacuation planner built with Python and Streamlit. The project compares A*, Breadth-First Search (BFS), and Greedy Best-First Search (GBFS) on grid-based evacuation scenarios with walls, fire hazards, dynamic replanning, fire-risk visualization, and a lightweight ML-style algorithm recommender.

## Features

- Interactive Streamlit grid editor for creating evacuation maps.
- Static pathfinding where fire cells are treated as blocked obstacles.
- Dynamic replanning when fire appears on the next planned step.
- A*, BFS, and GBFS implementations with matching dynamic variants.
- Algorithm comparison table showing path availability, steps, replans, and runtime.
- Trained recommender that suggests the best algorithm for the current grid.
- Fire-risk probability map based on proximity to active fire cells.
- Built-in presets for normal, fire-heavy, obstacle-heavy, BFS-friendly, and GBFS-friendly scenarios.

## Project Structure

```text
.
|-- Astar.py                  # A* search and dynamic A* replanning
|-- BFS.py                    # BFS search and dynamic BFS replanning
|-- GBFS.py                   # GBFS search and dynamic GBFS replanning
|-- GUI.py                    # Main A* Streamlit interface
|-- GUI_BFS.py                # BFS-focused Streamlit interface
|-- GUI_GBFS.py               # GBFS-focused Streamlit interface
|-- GUI_ML.py                 # ML recommendation and comparison Streamlit app
|-- train_mode.py             # Generates comparison data and trains recommender
|-- mode_model.pkl            # Saved recommender model
|-- comparison_results.pkl    # Saved generated comparison scenarios
|-- requirements.txt          # Python dependencies
`-- README.md
```

## Requirements

- Python 3.8 or higher
- Streamlit

Install dependencies:

```bash
pip install -r requirements.txt
```

## How to Run

Run the full ML recommendation app:

```bash
streamlit run GUI_ML.py
```

Then open the local Streamlit URL shown in the terminal, usually:

```text
http://localhost:8501
```

You can also run the individual algorithm demos:

```bash
streamlit run GUI.py
streamlit run GUI_BFS.py
streamlit run GUI_GBFS.py
```

## Training the Recommender

The repository already includes `mode_model.pkl`, but you can regenerate it at any time:

```bash
python train_mode.py
```

This script creates randomized grid scenarios, evaluates A*, BFS, and GBFS, selects the best algorithm using path length, replanning count, and runtime, then saves:

- `mode_model.pkl`
- `comparison_results.pkl`

## Grid Symbols

| Symbol | Meaning |
| ------ | ------- |
| `S` | Start position |
| `E` | Goal/exit position |
| `#` | Wall or blocked cell |
| `F` | Fire hazard |
| `.` | Empty walkable cell |
| Path highlight | Route found by the selected algorithm |

## Algorithms

### A*

A* uses both the cost already traveled and the Manhattan distance to the goal:

```text
f(n) = g(n) + h(n)
```

Where:

- `g(n)` is the path cost from the start to the current node.
- `h(n)` is the Manhattan distance to the goal.
- `f(n)` is the estimated total cost.

### BFS

BFS explores the grid level by level. Because every move has equal cost, BFS finds the shortest path in terms of number of steps when a path exists.

### GBFS

GBFS chooses the next cell using only the heuristic distance to the goal. It is often fast in open grids, but it does not guarantee the shortest path in all obstacle layouts.

## Dynamic Replanning

Dynamic mode simulates fire appearing during traversal:

1. The algorithm plans an initial path.
2. The agent follows the path step by step.
3. If the next step becomes a fire cell, that cell is marked blocked.
4. The algorithm replans from the current position.
5. The process continues until the goal is reached or no valid path remains.

The app reports the final path, number of steps, and total replans.

## ML Recommendation

`GUI_ML.py` extracts simple scenario features from the current grid:

- Number of rows
- Number of columns
- Wall count
- Fire count
- Obstacle density
- Manhattan distance from start to goal

The recommender uses these features to estimate whether A*, BFS, or GBFS is most suitable. The app also runs a live comparison on the current grid and shows the measured winner.

## Example Workflow

1. Start the app with `streamlit run GUI_ML.py`.
2. Load a preset or create a custom grid.
3. Place a start cell and a goal cell.
4. Add walls and fire hazards.
5. Choose `A* static` or `Dynamic` mode.
6. Click `Run Recommended` to follow the suggested algorithm.
7. Click `Compare All` to benchmark A*, BFS, and GBFS on the same grid.
8. Review the fire-risk probability map and comparison table.

## Complexity

| Algorithm | Time Complexity | Space Complexity |
| --------- | --------------- | ---------------- |
| A* | `O(E log V)` with a priority queue | `O(V)` |
| BFS | `O(V + E)` | `O(V)` |
| GBFS | `O(E log V)` with a priority queue | `O(V)` |

For dynamic replanning, the worst case is approximately:

```text
O(k * algorithm_cost)
```

Where `k` is the number of replanning events.

## Learning Outcomes

- Informed search using A* and Manhattan distance.
- Uninformed shortest-path search using BFS.
- Heuristic-first search using GBFS.
- Dynamic replanning in changing environments.
- Algorithm benchmarking and comparison.
- Lightweight model training with generated scenarios.
- Interactive visualization with Streamlit.

## License

Developed for academic and learning purposes.
