# test program for dataframe to tensor conversion
import pandas as pd
import numpy as np
import tensorflow as tf
import unittest

# Import your function here
from DataframeUtils import DataframeUtils

dataframeUtils: DataframeUtils
tensor_method=0

class TestDfToTensor(unittest.TestCase):
    def setUp(self):
        # Create a sample DataFrame
        self.df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'B': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            'C': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        })
        self.seq_len = 3

    def test_output_shape(self):
        tensor = dataframeUtils.df_to_tensor(self.df, self.seq_len, method=tensor_method)
        expected_shape = (self.df.shape[0] - self.seq_len + 1, self.seq_len, self.df.shape[1])
        self.assertEqual(tensor.shape, expected_shape)

    # def test_output_type(self):
    #     tensor = dataframeUtils.df_to_tensor(self.df, self.seq_len, method=tensor_method)
    #     self.assertIsInstance(tensor, tf.Tensor)

    def test_output_values(self):
        tensor = dataframeUtils.df_to_tensor(self.df, self.seq_len, method=tensor_method)
        # np_array = tensor.numpy() # type: ignore
        np_array = tensor # type: ignore

        # Check the first sequence
        expected_first_seq = np.array([
            [1, 10, 100],
            [2, 20, 200],
            [3, 30, 300]
        ])
        np.testing.assert_array_equal(np_array[0], expected_first_seq)

        # Check the last sequence
        expected_last_seq = np.array([
            [8, 80, 800],
            [9, 90, 900],
            [10, 100, 1000]
        ])
        np.testing.assert_array_equal(np_array[-1], expected_last_seq)

    # def test_invalid_seq_len(self):
    #     with self.assertRaises(ValueError):
    #         dataframeUtils.df_to_tensor(self.df, self.df.shape[0] + 1)

if __name__ == '__main__':
    dataframeUtils = DataframeUtils()
    tensor_method = 0
    unittest.main()