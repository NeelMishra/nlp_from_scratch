import torch


class nn_activations:
    def __init__(self):
        pass

    def binary_step_function(self, x : float, threshold : float) -> float:
        '''
        Arguments :-
            x : input vector
            threshold : threshold above which binary step function allows the input to pass
        Returns :- 
            x if the input is above threshold, else it will return 0
        Errors :-
            None
        '''

        if x > threshold:
            return 1
        else:
            return 0
        

        
    
        
    
        