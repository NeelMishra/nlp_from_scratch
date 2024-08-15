import unittest
from utils.activations import nn_activations

class test_activations_binary_step_function(unittest.TestCase):

    def setUp(self) -> None:
        self.testing_class = nn_activations()


    def test_binary_step_function_case_input_greater_than_threshold(self) -> None:
        """This test function will test the cases whe the input is greater than threshold"""
        threshold = 0.4
        input_x = 0.6

        output = self.testing_class.binary_step_function(threshold=threshold, x=input_x)
        expected = 1

        self.assertEqual(output, expected)

    def test_binary_step_function_case_input_smaller_than_threshold(self) -> None:
        """This test function will test the cases whe the input is lesser than threshold"""
        threshold = 0.4
        input_x = 0.3

        output = self.testing_class.binary_step_function(threshold=threshold, x=input_x)
        expected = 0

        self.assertEqual(output, expected)

    def test_binary_step_function_case_input_exactly_equal_to_threshold(self) -> None:
        """This test function will test the cases whe the input is lesser than threshold"""
        threshold = 0.4
        input_x = 0.4

        output = self.testing_class.binary_step_function(threshold=threshold, x=input_x)
        expected = 0

        self.assertEqual(output, expected)
