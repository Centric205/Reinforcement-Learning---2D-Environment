from FourRooms import FourRooms
import numpy as np
from Q_learning import *
import sys


def main():
    print("File name: ", sys.argv[0])
    print("Your CLM command is ", sys.argv[1])
    # Creating Q table initialised to 0
    Q_s_a = np.zeros((13*13, 4))

    # Create FourRooms Object
    fourRoomsObj = FourRooms('simple')
    actSeq = [FourRooms.UP,FourRooms.RIGHT, FourRooms.DOWN, FourRooms.LEFT]


    fourRoomsObj.showPath(-1)


if __name__ == "__main__":
    main()
