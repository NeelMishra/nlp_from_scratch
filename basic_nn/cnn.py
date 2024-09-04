import numpy as np
from typing import Tuple

class CNN:
    def __init__(self):
        pass

    def rotate_90_clockwise(self, X:np.ndarray) -> np.ndarray:
        return np.flip(X.T, axis=1) # The logic is same as leetcode question 48. Basically you take the transpose and what you observe is that the desired output is same as the inverted rows, np.flip will do the inversion of rows.

    def rotate_180_clockwise(self, X:np.ndarray):
        return self.rotate_90_clockwise(self.rotate_90_clockwise(X))
    def padding_operation(self, X : np.ndarray, pad_size : int = 0) -> np.ndarray:

        input_h, input_w, input_c = X.shape
        output_array = np.zeros((input_h + 2 * pad_size, input_w + 2 * pad_size, input_c))

        for h in range(input_h):
            for w in range(input_w):
                for c in range(input_c):
                    output_array[h+pad_size, w+pad_size, c] = X[h, w, c]
        
        return output_array


    def convolution_operation(self, X: np.ndarray, F: np.ndarray) -> float:
        """
        Perform convolution operation between an input patch and a filter.

        Args:
        X : Input image patch (Channel, height, width)
        F : Filter (Channel, height, width)

        Returns:
        float: Result of the convolution operation
        """
        return np.sum(X * F)

    def convolution(self, X: np.ndarray, F: np.ndarray, stride: int, conv_type='basic') -> np.ndarray:
        """
        Perform convolution on the input X using filter F.

        Args:
        X : Input image (Height, Width, Channel)
        F : Filters (Number of filters, Height, Width, Channel)
        stride : Stride for the convolution

        Returns:
        np.ndarray: Output of the convolutional layer
        """

        num_filters, filter_h, filter_w, filter_c = F.shape

        if conv_type == 'full':
            X = self.padding_operation(X, pad_size=filter_h-1)

        image_h, image_w, image_c = X.shape
        

        assert image_c == filter_c, "Input and filter channels must match"

        output_h = (image_h - filter_h) // stride + 1
        output_w = (image_w - filter_w) // stride + 1

        output = np.zeros((num_filters, output_h, output_w))

        for f in range(num_filters):
            for i in range(0, output_h):
                for j in range(0, output_w):
                    h_start, h_end = i * stride, i * stride + filter_h
                    w_start, w_end = j * stride, j * stride + filter_w
                    input_slice = X[h_start:h_end, w_start:w_end, :]
                    output[f, i, j] = self.convolution_operation(input_slice, F[f])

        return output

    def feedforward(self):
        pass

    def backpropagate(self):
        pass


if __name__ == '__main__':
    model = CNN()
    test_image = np.array([
        [1,6,2],
        [5,3,1],
        [7,0,4],
    ]).reshape(3,3,1)
    test_filter = np.array([
        [
            [1,2],
            [-1,0]
        ],
        [
            [1,2],
            [-1,0]
        ]
    ]).reshape(2,2,2,1)


    # convolution_result = model.convolution(X = test_image, F = test_filter, stride=1, conv_type='full')

    # print(convolution_result)

    # padding_result = model.padding_operation(test_image, pad_size=2)

    # print(padding_result.shape)

    # print(padding_result.reshape(padding_result.shape[0],padding_result.shape[1]))

    dummy_input = np.array([[1,2,3], [4,5,6], [7,8,9]])
    print(model.rotate_90_clockwise(dummy_input))
    print(model.rotate_180_clockwise(dummy_input))