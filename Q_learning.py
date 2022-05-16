

def rewards(grid_type):
    if grid_type == 0:
        return -1

    elif grid_type == 1:
        return 100

    elif grid_type == 2:
        return 400

    elif grid_type == 3:
        return 600

def rewards_3(cell_type, packages):

    # Positive vibes for Red (^.^))
    if cell_type == 1:
        packages[0] = True
        return 300

    # Reward if Red is already picked
    elif cell_type == 2 and packages[1]:
        packages[2] = True
        return 200

    elif cell_type == 2 and not packages[1]:
        return -2

    # +(ive) reward if both red and  green have been picked up
    elif cell_type == 3 and packages[1] and packages[2]:
        packages[3] = True
        return 100
    elif cell_type == 3 and not packages[1] and not packages[2]:
        return -2
    # -(ive) reward for empty and packages picked in the wrong order
    else:
        return -1

