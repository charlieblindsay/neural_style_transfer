import streamlit as st
import tensorflow as tf
import numpy as np
import PIL

from utils import *

st.title('Neural Style Transfer Demo')
st.write('After the user has uploaded a content image and a style image, an image with the \'content\' of the content image and the \'style\' of style image is generated.')

st.subheader('File upload')

st.write(np.__version__)
st.write(PIL.__version__)
st.write(tf.__version__)
st.write(st.__version__)