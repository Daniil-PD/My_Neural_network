import random
import math

def sigmoid(gamma):
  if gamma < 0:
    return 1 - 1/(1 + math.exp(gamma))
  else:
    return 1/(1 + math.exp(-gamma))

class Neuron():
    def __init__(self, input_shape:int = None, weights:list[float] = None, bias:float = None, type_activation_function = "sigmoid"):
        if weights is None and bias is None and not input_shape is None:
            self.input_shape = input_shape
            self.generate_randomize_weights()
            self.bias = random.random()*2-1
        elif not weights is None and not bias is None:
            self.input_shape = len(weights)
            self.weights = weights
            self.bias = bias

        else:
            raise ValueError

        self.input = None
        self.weights_change = [0]*self.input_shape
        self.bias_change = 0
        self.activation = None
        self.weighted_sum_input = None
        self.type_activation_function = type_activation_function
        
    def forward(self, input:list[float]) -> float:
        if len(input) != self.input_shape:
            raise ValueError
        
        self.input = input
        
        self.weighted_sum_input = 0
        for i in range(self.input_shape):
            self.weighted_sum_input += self.weights[i]*input[i]
        self.weighted_sum_input += self.bias
        self.activation = self.activation_function(self.weighted_sum_input)

        return self.activation
    
    def activation_function(self, input):
        if self.type_activation_function == "sigmoid":
            # try: 
            #sigmoid
                return sigmoid(input)
            # except OverflowError:
            #     print(input)
            #     raise OverflowError
        raise ValueError(f"Invalid activation function {self.type_activation_function}")
    
    def delta_activation_function(self, input):
        if self.type_activation_function == "sigmoid": 
            #sigmoid
            return (sigmoid(input))*(sigmoid(-input))
        raise ValueError(f"Invalid activation function {self.type_activation_function}")

        
    def generate_randomize_weights(self):
        self.weights = []
        for i in range(self.input_shape):
            self.weights.append(random.random()*2-1)


    def backward(self, output_error:float, learning_rate:float = 0.1) -> list[float]:
        if self.input is None:
            raise RuntimeError("Use forward method first")
        
        input_error = [0]*self.input_shape

        temp_optimization = -learning_rate*self.delta_activation_function(self.weighted_sum_input)*output_error
        for i in range(self.input_shape):
            # Умножение следующих:
            # - значение производной активационной функции в точке взвешенной суммы входных сигналов 
            # - коэффициента обучения
            # - ошибки выхода (отличие от ожидаемого значения)
            # - входной сигнал

            self.weights_change[i] += temp_optimization*self.input[i]

            input_error[i] = output_error*self.weights[i]

        # Если предположить что смещение это коэффициент для входа всегда равного 1
        self.bias_change += temp_optimization

        return input_error



    def update_weights(self, learning_rate:float = 1):
        for i in range(self.input_shape):
            self.weights[i] += self.weights_change[i]*learning_rate
            self.weights_change[i] = 0
        self.bias += self.bias_change*learning_rate
        self.bias_change = 0
