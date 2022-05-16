from FourRooms import FourRooms
import numpy as np
from Q_learning import *
import sys


def main():
    # Creating Q table initialised to 0
    Q_s_a = np.zeros((13 * 13, 4))

    # CLM argument
    isStochastic = None
    if len(sys.argv) > 1:
        if sys.argv[1] == "-stochastic":
            isStochastic = True

    # Create FourRooms Object
    fourRoomsObj = FourRooms('rgb', isStochastic)
    actSeq = [FourRooms.UP, FourRooms.RIGHT, FourRooms.DOWN, FourRooms.LEFT]

    # Hyper-parameters
    epsilon = 0.1
    discount_factor = 0.6
    learning_rate = 0.1

    for i in range(10000):
        fourRoomsObj.newEpoch()
        x, y = fourRoomsObj.getPosition()
        state = x * 13 + y

        isTerminal = False
        packages = {1: False, 2: False, 3: False}

        while not isTerminal:
            if np.random.uniform(0, 1) < epsilon:
                action = actSeq[np.random.randint(4)]
            else:
                action = np.argmax(Q_s_a[state])

            cell_type, (new_X, new_Y), numpack, isTerminal = fourRoomsObj.takeAction(action)

            old_q_value = Q_s_a[state, action]
            newstate = new_X * 13 + new_Y
            nextmax = np.max(Q_s_a[newstate])

            TP = rewards_3(cell_type,packages) + (discount_factor * nextmax) - old_q_value
            newValue = old_q_value + (learning_rate * TP)
            Q_s_a[state, action] = newValue

            state = newstate

            if isTerminal and not packages[1]:
                fourRoomsObj.newEpoch()
            else:
                if isTerminal and not packages[2]:
                    fourRoomsObj.newEpoch()

            if packages[1] and packages[2] and packages[3]:
                print("DONE....")
                isTerminal = True

    print(Q_s_a)
    print("Training finished on epoch: ", i)
    fourRoomsObj.showPath(-1)

    # Displays the optimal path
    Done = False
    while not Done:
        x, y = fourRoomsObj.getPosition()
        state = x * 13 + y
        action = np.argmax(Q_s_a[state])
        cell_type, (new_X, new_Y), numpack, isTerminal = fourRoomsObj.takeAction(action)

        newstate = new_X * 13 + new_Y
        nextmax = np.max(Q_s_a[newstate])
        state = nextmax

        if numpack == 0:
            Done = True



    fourRoomsObj.showPath(-1)


if __name__ == "__main__":
    main()