from utils.activations import nn_activations
from typing import List, Tuple, Any
import numpy as np
import numpy.typing as npt
class ANN:

    def __init__(self, dimensions: List[Tuple[int, int, str]]):
        '''
        This is the constructor function that is responsible to set the ANN's weights, and initialize other class members
        '''

        if dimensions == []:
            raise SyntaxError("Artificial neural network cannot be initialized to empty weights!")
        
        self.layers = []

        prev_dimension = None

        for dimension in dimensions:
            activation_string = dimension[-1]
            if activation_string == 'sigmoid':
                activation = nn_activations.sigmoid
            elif activation_string == 'binary_step':
                activation = nn_activations.binary_step_function
            else:
                raise NotImplementedError(f"Activation fuction {activation_string} not yet implemented!")
            
            
            if prev_dimension != None and prev_dimension[1] != dimension[0]:
                    raise SyntaxError(f"The dimensions are mismatching for the layer {prev_dimension} x {dimension}, basically {prev_dimension[1]} != {dimension[0]}")

            self.layers.append(
                {
                    'weight' : self.weight_init(dimension),
                    'bias' : self.weight_init((1, dimension[-2])),
                    'activation' : activation
                }
            )

            prev_dimension = dimension

    def weight_init(self, dimension : Tuple[int, int]) -> npt.NDArray[Any]:
        '''
        This function is responsible to randomly initialize, and return the weight matrices.

        Args:
        dimension : Tuple of ints which specifies the dimension of the about to be initialized weight vector

        Returns:
        Randomly initialized numpy array of dimensions equal to the dimensions in the argument.

        Exceptions:
        None
        '''
        return np.random.randn(dimension[0], dimension[1])
    
    def feed_forward(self, X : npt.NDArray[Any]) -> npt.NDArray[Any]:
        '''
        This function is responsible to do the feed forward propogation in the artifical neural network
        step 1 : Calculate the matrix product of the past input with the current weight (edges of each past node to the current nodes) and it with the bias
        step 2 : Pass it to the non linear activation function
        '''
        accumilator = X
        for idx, layer in enumerate(self.layers):
            '''if the past activation dimensions are m x n and there are o next nodes,
               the weight dimension will be n x o, and the past activation dimension is m x n
               hence the multiplication should be past_activation x weight => m x o, where m is hopefully the batchsize

               when the bias is added, bias dimension is o x 1, this is expanded m times (vertically).
               [bias_11 bias_12 bias_13 bias_14 bias_15] becomes

               [bias_11 bias_12 bias_13 bias_14 bias_15]
               [bias_11 bias_12 bias_13 bias_14 bias_15]
               [bias_11 bias_12 bias_13 bias_14 bias_15]
               [bias_11 bias_12 bias_13 bias_14 bias_15]
               ..
               ..
               m times
            '''
            accumilator = np.matmul(accumilator, layer['weight']) + layer['bias'] 
            accumilator = layer['activation'](x=accumilator, backward=False)

        return accumilator
    
    
    


if __name__=='__main__':
    ann = ANN([(2,3, 'sigmoid'), (3,5, 'sigmoid'), (5, 1, 'sigmoid')])
    
    m = 100
    dummy_input = np.ones((m, 2))
    output = ann.feed_forward(dummy_input)

    print(f"Output of the feedforward has the shape {output.shape}")
