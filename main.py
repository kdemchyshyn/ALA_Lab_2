import Task_1 as t1
import Task_2 as t2
import Task_3 as t3
import numpy as np

def main():
    # Task 1
    matrix = np.array([[-5, 0, 3], [-6, 1 ,3], [-6, 0, 4]])
    eigen = t1.getEigen(matrix)

    for el in eigen:
        print(f"{el[0]} : {el[1]}")

    # Task 2
    t2.compressImage()

    # Task 3
    message = "Hello, grey rainy world. When will we see snow?"
    t3.task3(message)
    return 0

if __name__ == '__main__':
    main()