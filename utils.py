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

def vgg_layers(layer_names):
  """ Creates a VGG model that returns a list of intermediate output values."""
  # Load our model. Load pretrained VGG, trained on ImageNet data
  vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
  vgg.trainable = False

  outputs = [vgg.get_layer(name).output for name in layer_names]

  model = tf.keras.Model([vgg.input], outputs)
  return model

def gram_matrix(input_tensor):
  result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
  input_shape = tf.shape(input_tensor)
  num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
  return result/(num_locations)


def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


class StyleContentModel(tf.keras.models.Model):
  def __init__(self, style_layers, content_layers):
    super(StyleContentModel, self).__init__()
    self.vgg = vgg_layers(style_layers + content_layers)
    self.style_layers = style_layers
    self.content_layers = content_layers
    self.num_style_layers = len(style_layers)
    self.vgg.trainable = False

  def call(self, inputs):
    "Expects float input in [0,1]"
    inputs = inputs*255.0
    preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
    outputs = self.vgg(preprocessed_input)
    style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                      outputs[self.num_style_layers:])

    style_outputs = [gram_matrix(style_output)
                     for style_output in style_outputs]

    content_dict = {content_name: value
                    for content_name, value
                    in zip(self.content_layers, content_outputs)}

    style_dict = {style_name: value
                  for style_name, value
                  in zip(self.style_layers, style_outputs)}

    return {'content': content_dict, 'style': style_dict}
  
