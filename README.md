# Smart Emergency Evacuation Planner

Dynamic A\* Search · Manhattan Heuristic · Fire Replanning Simulation ·
Streamlit GUI

------------------------------------------------------------------------

## 1. Project Overview

The Smart Emergency Evacuation Planner is a grid-based pathfinding
system built using the A\* search algorithm.

It supports:

• Static A\* search (fire treated as obstacle)\
• Dynamic A\* search (fire discovered during traversal and triggers
replanning)

The system includes an interactive Streamlit GUI and predefined test
cases for demonstration.

------------------------------------------------------------------------

## 2. Execution Instructions

### Requirements

Python 3.8 or higher

### Install Dependencies

Using requirements.txt:

    pip install -r requirements.txt

Or manually:

    pip install streamlit

(Standard libraries used: heapq, copy)

------------------------------------------------------------------------

## 3. How to Run

### Run GUI Version

    streamlit run GUI.py

Then open in browser:

    http://localhost:8501

------------------------------------------------------------------------

### Run Algorithm Only (CLI Testing)

If using separate test files:

    python testCase.py

------------------------------------------------------------------------

## 4. Preset Test Cases Included

The GUI contains the following preset scenarios:

TC1 --- Normal A\* - Standard grid with walls - Static mode

TC2 --- Fire Avoidance - Fire treated as static obstacle - Static mode

TC3A --- Dynamic Replan (fire @1,0) - Fire discovered during traversal -
Dynamic mode

TC3B --- Dynamic Replan (fire @1,1) - Different fire location - Dynamic
mode

TC4 --- No Possible Path - Goal completely blocked - Static mode

Custom (blank grid) - User-defined grid

------------------------------------------------------------------------

## 5. Sample Input / Output Demonstrations

### Example 1 --- TC1 Normal A\*

Grid:

S . . . \# \# . \# . . . E

Mode: Static A\*

Output:

Path Found\
Steps: 5\
Replans: 0

Path: (0,0) → (0,1) → (0,2) → (1,2) → (2,2) → (2,3)

------------------------------------------------------------------------

### Example 2 --- TC3A Dynamic Replan

Grid (fire at 1,0 discovered during traversal):

S . . . . . . . . . . E

Hidden Fire: (1,0)

Mode: Dynamic A\*

Output:

Path Found\
Steps: 6\
Replans: 1

Path (after replanning): (0,0) → (0,1) → (0,2) → (1,2) → (2,2) → (2,3)

------------------------------------------------------------------------

### Example 3 --- TC4 No Possible Path

Grid:

S \# \# \# \# . . . \# . . E

Output:

No Path Found\
Steps: 0\
Replans: 0

------------------------------------------------------------------------

## 6. Technical Implementation Overview

### A\* Algorithm

The A\* search algorithm selects nodes based on:

    f(n) = g(n) + h(n)

Where:

g(n) = cost from start to node\
h(n) = Manhattan heuristic\
f(n) = estimated total cost

Heuristic used:

    h(a,b) = |row1 - row2| + |col1 - col2|

Movement: 4-directional (up, down, left, right)\
Cost per move: 1

Priority queue implementation: heapq\
Best g-cost tracking: dictionary (best_g)

------------------------------------------------------------------------

### Dynamic A\* Strategy

1.  Compute initial path ignoring new fire cells.
2.  Traverse path step-by-step.
3.  If next step becomes fire: • Mark cell as blocked\
    • Increment replan counter\
    • Re-run A\* from current position\
4.  Continue until goal reached or no path exists.

Final output includes: • Complete executed path\
• Total replanning count

------------------------------------------------------------------------

## 7. Complexity Analysis

Static A\*: Time: O(E log V)\
Space: O(V)

Dynamic Mode: Worst Case: O(k × A\*)\
Where k = number of replans

------------------------------------------------------------------------

## 8. Grid Symbols

S → Start\
E → Goal\
\# → Wall (blocked)\
F → Fire\
. → Empty cell

------------------------------------------------------------------------

## 9. Learning Outcomes

• Implementation of informed search (A\*)\
• Heuristic design and admissibility\
• Real-time replanning in dynamic environments\
• Use of priority queues\
• Interactive visualization with Streamlit

------------------------------------------------------------------------

## 10. License

Developed for academic purposes.
