import heapq
import copy

# Node class for GBFS algorithm
class Node:
    def __init__(self, position, parent=None, h=0):
        self.position = position
        self.parent = parent
        self.h = h  # GBFS only uses h(n) 

    def __lt__(self, other):
        return self.h < other.h  # heap ordered purely by heuristic h(n)

# Heuristic: Manhattan Distance
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# GBFS algorithm
def gbfs(grid, start, goal):
    rows, cols = len(grid), len(grid[0])
    open_list = []
    visited = set()
    visited.add(start)

    heapq.heappush(open_list, Node(start, None, heuristic(start, goal)))

    while open_list:
        current = heapq.heappop(open_list)

        if current.position == goal:
            return reconstruct_path(current)

        # 4-way movement: Up, Down, Left, Right
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for dr, dc in moves:
            r = current.position[0] + dr
            c = current.position[1] + dc
            next_pos = (r, c)

            # Boundary check
            if 0 <= r < rows and 0 <= c < cols:
                # Obstacle/Fire check
                if grid[r][c] in ['#', 'F']:
                    continue

                if next_pos not in visited:
                    visited.add(next_pos)
                    h = heuristic(next_pos, goal)
                    heapq.heappush(open_list, Node(next_pos, current, h))

    return None

def reconstruct_path(node):
    path = []
    while node:
        path.append(node.position)
        node = node.parent
    return path[::-1]

def dynamic_gbfs(grid, start, goal, new_fire_cells):
    grid = copy.deepcopy(grid)
    current_position = start
    fires = list(new_fire_cells)
    replan_count = 0
    current_path = gbfs(grid, current_position, goal)

    # If no path exists at the start, return immediately
    if not current_path:
        return None, 0

    final_executed_path = [current_position]
    # i starts at 1 because current_position is already the first step in the path
    i = 1
    while i < len(current_path):
        next_step = current_path[i]

        # Check if the next step has become a fire cell
        if next_step in fires:
            r, c = next_step
            grid[r][c] = 'F'
            fires.remove(next_step)
            replan_count += 1
            # Replan from the current position to the goal with the updated grid
            new_plan = gbfs(grid, current_position, goal)

            if not new_plan:
                return None, replan_count

            current_path = new_plan
            i = 1
            continue

        # Move to the next step
        current_position = next_step
        final_executed_path.append(current_position)
        i += 1

    return final_executed_path, replan_count
