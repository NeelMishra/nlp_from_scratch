'''
This module is not concerned with parrallelism, but instead concerned with implemented the bare basics
'''
import numpy as np

class nn_activations:
    def __init__(self):
        pass
    
    def binary_step_function( x : float, threshold : float, backward : bool = False) -> float:
        '''
        Arguments :-
            x : input vector
            threshold : threshold above which binary step function allows the input to pass
            backward : flag that indicates if the gradient of the function must be returned for backpropogation/backward pass
        Returns :- 
            x if the input is above threshold, else it will return 0
        Errors :-
            None
        '''

        if backward:
            return 0

        else:
            if x > threshold:
                return 1
            else:
                return 0
        
    def sigmoid( x : float, backward : bool = False) -> float:
        '''
        Arguments :-
            x : input vector
            backward : flag that indicates if the gradient of the function must be returned for backpropogation/backward pass
        Returns :-
            The sigmoid of x (floating value) between [0, 1]
        Errors :-
        '''
        if backward:
            cached_activation = nn_activations.sigmoid(x, backward=False)
            return cached_activation * (1-cached_activation)

        else:
            return 1/(1 + np.exp(-x))
    
        

        
    
        
    
        