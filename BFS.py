from collections import deque
import copy

# BFS algorithm
def bfs(grid, start, goal, predicted_fires=None):
    if predicted_fires is None:
        predicted_fires = []
    if isinstance(predicted_fires, tuple) and len(predicted_fires) == 2 and isinstance(predicted_fires[0], int):
        predicted_fires = [predicted_fires]
    predicted_set = set(predicted_fires)
    rows, cols = len(grid), len(grid[0])

    # Queue stores (position, path_so_far)
    queue = deque()
    deferred_queue = deque()
    queue.append((start, [start]))

    # Visited set to avoid revisiting nodes
    visited = set()
    visited.add(start)

    while queue or deferred_queue:
        if queue:
            current_pos, path = queue.popleft()
        else:
            current_pos, path = deferred_queue.popleft()

        if current_pos == goal:
            return path

        # 4-way movement: Up, Down, Left, Right (all cost 1)
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for dr, dc in moves:
            r = current_pos[0] + dr
            c = current_pos[1] + dc
            next_pos = (r, c)

            # Boundary check
            if 0 <= r < rows and 0 <= c < cols:
                # Obstacle/Fire check
                if grid[r][c] in ['#', 'F']:
                    continue

                if next_pos not in visited:
                    visited.add(next_pos)
                    if next_pos in predicted_set:
                        deferred_queue.append((next_pos, path + [next_pos]))
                    else:
                        queue.append((next_pos, path + [next_pos]))

    return None


# Dynamic Replanning Function (mirrors dynamic_astar structure)
def dynamic_bfs(grid, start, goal, new_fire_cells, predicted_fires=None):
    grid = copy.deepcopy(grid)
    current_position = start
    fires = list(new_fire_cells)
    replan_count = 0
    current_path = bfs(grid, current_position, goal, predicted_fires)

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
            new_plan = bfs(grid, current_position, goal, predicted_fires)

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
