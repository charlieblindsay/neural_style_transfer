import unittest
import utils
import tensorflow as tf
import numpy as np

class TestUtils(unittest.TestCase):

    def test_load_img_tensor_from_path(self):
        """Function which tests whether the image_tensor generated using load_img_tensor_from_path function
        is the equal to the actual image tensor.
        """
        test_image_tensor = utils.load_img_tensor_from_path('./tests/test_image.png')
        actual_test_image_tensor = tf.convert_to_tensor(np.load('./tests/actual_test_image_tensor.npy'), dtype=tf.float32)
        self.assertTrue(tf.math.reduce_all(tf.equal(test_image_tensor,actual_test_image_tensor)).numpy())

if __name__ == '__main__':
    unittest.main()