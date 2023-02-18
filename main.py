import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np

from utils import *

# Load content and style images (see example in the attached colab).
content_image = load_img('./content.png')
style_image = load_img('./style.png')
# Convert to float32 numpy array, add batch dimension, and normalize to range [0, 1]. Example using numpy:
# Optionally resize the images. It is recommended that the style image is about
# 256 pixels (this size was used when training the style transfer network).
# The content image can be any size.

# Load image stylization module.
hub_module = tf.saved_model.load.load('model.pb')

# Stylize image.
outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
stylized_image = outputs[0]
