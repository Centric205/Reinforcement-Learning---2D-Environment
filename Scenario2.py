from FourRooms import FourRooms
import numpy as np
from Q_learning import *


def main():
    # Create FourRooms Object
    Q_s_a = np.zeros((13 * 13, 4))
    fourRoomsObj = FourRooms('multi')
    actSeq = [FourRooms.UP,FourRooms.RIGHT, FourRooms.DOWN, FourRooms.LEFT]


    epsilon = 0.1
    discount_factor = 0.6
    learning_rate = 0.1

    for i in range(10000):
        fourRoomsObj.newEpoch()
        x, y = fourRoomsObj.getPosition()
        state = x * 13 + y
        isTerminal = False

        while not isTerminal:
            if np.random.uniform(0, 1) < epsilon:
                action = actSeq[np.random.randint(4)]
            else:
                action = np.argmax(Q_s_a[state])

            cell_type, (new_X,new_Y), numpack, isTerminal = fourRoomsObj.takeAction(action)
            old_q_value = Q_s_a[state, action]
            newstate = new_X*13 + new_Y
            nextmax = np.max(Q_s_a[newstate])

           # new_value = (1 - learning_rate) * old_q_value + learning_rate * (rewards(cell_type) + discount_factor * nextmax)
           # Q_s_a[state, action] = new_value

            TP = rewards(cell_type) + (discount_factor * nextmax) - old_q_value
            newValue = old_q_value + (learning_rate * TP)
            Q_s_a[state, action] = newValue
           # isTerminal = False
           # print(i)
            state = newstate
           # if numpack == 0:
           #     isTerminal = True

    print(Q_s_a)
    print("Training finished on epoch: ", i)      
    fourRoomsObj.showPath(-1)


if __name__ == "__main__":
    main()