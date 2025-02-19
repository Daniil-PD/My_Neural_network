from my_NN_framework import Neral_network
from my_NN_framework.layer import Input_layer, NN_layer


            


if __name__ == "__main__":
    nn = Neral_network([3, 6, 8])
    x = [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1]
    ]

    y = [
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
    ]


    nn.fit(x, y, epochs=10000, learning_rate=0.5)



    for i_x, i_y in zip(x, y):
        output = nn.forward(i_x)
        print(f"input: {i_x}, expected: {i_y}, output: ", end="")
        for i in output:
            print(f"{i:.2}", end=" ")
        print()

