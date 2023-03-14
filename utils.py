""" Contains utility functions used in app.py:
- load_img_tensor_from_path
- convert_tensor_to_img
"""

import tensorflow as tf
import numpy as np
import PIL

def load_img_tensor_from_path(path_to_img: str):
  """Returns a tensor of the image at the inputted file path.
  The tensor has the correct dimensions for being an input into the VGG-19 model.

  Args:
      path_to_img (str): File path to image

  Returns:
      tf.Tensor: Image tensor
  """
  max_dim = 512
  img_tensor = tf.io.read_file(path_to_img)
  img_tensor = tf.image.decode_image(img_tensor, channels=3)
  img_tensor = tf.image.convert_image_dtype(img_tensor, tf.float32)

  shape = tf.cast(tf.shape(img_tensor)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  img_tensor = tf.image.resize(img_tensor, new_shape)
  img_tensor = img_tensor[tf.newaxis, :]
  return img_tensor

def convert_tensor_to_img(tensor):
  """ Scales tensor values and converts tensor into an image

  Args:
      tensor (tf.Tensor): Image tensor

  Returns:
      Image object
  """
  tensor = tensor*255 # Scales tensor values from 0-1 to 0-255 to represent image
  tensor = np.array(tensor, dtype=np.uint8)

  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]

  return PIL.Image.fromarray(tensor)

# @st.cache_resource # saves the output
def load_model(path: str):
    """Loads a SavedModel from the folder at the inputted file path.
    A SavedModel contains both trained parameters and the computation,
    see https://www.tensorflow.org/guide/saved_model

    Args:
        path (str): Path to the SavedModel

    Returns:
        tf.saved_model: The SavedModel object
    """

    return tf.saved_model.load('model')

def make_grid(cols,rows):
    grid = [0]*cols
    for i in range(cols):
        with st.container():
            grid[i] = st.columns(rows)
    return grid