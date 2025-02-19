from abc import ABC, abstractmethod
from my_NN_framework.neuron import Neuron

class Layer(ABC):
    @abstractmethod
    def __init__(self, input_shape:int, neurons:int):
        pass

    @property
    @abstractmethod
    def activations(self):
        pass

class Input_layer(Layer):
    def __init__(self, output_shape:int):
        self.output_shape = output_shape
        self._activations = [0]*output_shape

    @property
    def activations(self):
        return self._activations
    
    @activations.setter
    def activations(self, activations:list[float]):
        if len(activations) != self.output_shape:
            raise ValueError
        self._activations = activations

class NN_layer(Layer):
    def __init__(self, input_shape:int, neurons:int):
        self.input_shape = input_shape
        self.output_shape = neurons
        self.neurons = []

        for i in range(neurons):
            self.neurons.append(Neuron(input_shape = input_shape))
    
    @property
    def activations(self):
        ret_list = []
        for neuron in self.neurons:
            ret_list.append(neuron.activation)
        return ret_list
    
    @activations.setter
    def activations(self, activations:list[float]):
        if len(activations) != self.output_shape:
            raise ValueError
        for i in range(len(activations)):
            self.neurons[i].activation = activations[i]

    def forward(self, input:list[float]) -> list[float]:
        if len(input) != self.input_shape:
            raise ValueError(f"input_shape({self.input_shape}) != len(input)({len(input)})")

        for i in range(len(self.neurons)):
            self.neurons[i].forward(input)

        return self.activations
        
    def backward(self, output_error:list[float], learning_rate:float = 0.1) -> list[float]:

        input_error = [0]*self.input_shape

        for i in range(len(self.neurons)):
            input_error_one_neuron = self.neurons[i].backward(output_error[i], learning_rate)
            for input_index in range(len(input_error_one_neuron)):
                input_error[input_index] += input_error_one_neuron[input_index]

        return input_error

    def update_weights(self, learning_rate:float = 0.1):
        for neuron in self.neurons:
            neuron.update_weights(learning_rate)
            