#from Astar import astar, dynamic_astar

# Test Case 1: Normal A*
grid1 = [
    ['S', '.', '.', '.'],
    ['#', '#', '.', '#'],
    ['.', '.', '.', 'E']
]

start = (0, 0)
goal = (2, 3)

print("Test Case 1 (A*):")
print(astar(grid1, start, goal))
print("-" * 40)


# Test Case 2: Fire Hazard Avoidance
grid2 = [
    ['S', '.', 'F', '.'],
    ['.', '#', 'F', '.'],
    ['.', '.', '.', 'E']
]

print("Test Case 2 (A* with Fire):")
print(astar(grid2, start, goal))
print("-" * 40)


# Test Case 3: Dynamic Replanning
grid3 = [
    ['S', '.', '.', '.'],
    ['.', '.', '.', '.'],
    ['.', '.', '.', 'E']
]

dynamic_fire = [(1, 0)]

print("Test Case 3 (Dynamic A* Replanning):")
print(dynamic_astar(grid3, start, goal,dynamic_fire))
print("-" * 40)
dynamic_fire = [(1, 1)]

print("Again with different fire cell:")
print(dynamic_astar(grid3, start, goal,dynamic_fire))
print("-" * 40)

# Test Case 4: No Possible Path
grid4 = [
    ['S', '#', '#', '#'],
    ['#', '.', '.', '.'],
    ['#', '.', '.', 'E']
]
print("Test Case 4 (Failure Case):")
print(astar(grid4, start, goal))
