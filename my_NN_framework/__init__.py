from my_NN_framework.layer import NN_layer, Input_layer

class Neral_network():
    def __init__(self, struct:list[int]):
        self.layers = []
        for index, neurons_count in enumerate(struct):
            if index == 0:
                self.layers.append(Input_layer(neurons_count))
            else:
                self.layers.append(NN_layer(struct[index-1], neurons_count))
        self.input_shape = struct[0]
        self.output_shape = struct[-1]

    def forward(self, input:list[float]) -> list[float]:
        if len(input) != self.input_shape:
            raise ValueError

        self.layers[0].activations = input

        for index_layer in range(0, len(self.layers)-1):
            self.layers[index_layer+1].forward(self.layers[index_layer].activations)

        return self.layers[-1].activations
    

    
    def fit(self, x:list[float], y:list[float], epochs:int = 100, learning_rate:float = 0.1):
        for epoch in range(epochs):
            sum_loss = 0
            for i in range(len(x)):
                self.forward(x[i])

                loss = 0
                for j in range(len(self.layers[-1].neurons)):
                    loss += (self.layers[-1].neurons[j].activation - y[i][j])**2
                sum_loss += loss
                
                error_last_layer = []
                for j in range(len(self.layers[-1].neurons)):
                    error_last_layer.append(self.layers[-1].neurons[j].activation - y[i][j])

                for layer_index in range(len(self.layers)-1, 0, -1):
                    error_last_layer = self.layers[layer_index].backward(error_last_layer, learning_rate)

                for layer_index in range(1, len(self.layers)):
                    self.layers[layer_index].update_weights()

                print(f"epoch: {epoch}({100*(i/len(x)):.3}%), loss: {loss:.6f}", end="\r")
            mean_loss = sum_loss/len(x)
            print(f"epoch {epoch}: sum_loss = {sum_loss:.6f}, mean_loss = {mean_loss:.6f}")
                
    
