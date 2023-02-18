import streamlit as st
import tensorflow as tf
import numpy as np
import PIL

from utils import *

@st.cache # saves the output
def load_model():
    return tf.saved_model.load('model')


def save_uploaded_files(uploaded_files: list):
    """Writes 

    Args:
        uploaded_files (_type_): _description_
    """
    for file in uploaded_files:
        file_name = file['type']
        file_name_with_extension = f'{file_name}.png'
        with open(file_name_with_extension,"wb") as f:
            f.write(file['img'].getbuffer())

st.title('Neural Style Transfer Demo')
st.write('After the user has uploaded a content image and a style image, an image with the \'content\' of the content image and the \'style\' of style image is generated.')

st.subheader('File upload')

st.write(np.__version__)
st.write(PIL.__version__)
st.write(tf.__version__)
st.write(st.__version__)




content_image = st.file_uploader("Choose a content image. This should be a photo.")
style_image = st.file_uploader('Choose a style image. This should be a piece of art, e.g. a painting')

if content_image is not None and style_image is not None:
    with st.spinner('Generating Image. This make take a while...'):
        save_uploaded_files([{'type': 'content', 'img': content_image}, {'type': 'style', 'img': style_image}])

        content_image = load_img('./content.png')
        style_image = load_img('./style.png')

        hub_model = load_model()
        stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]
        generated_image = tensor_to_image(stylized_image)

        content_image = tensor_to_image(content_image)
        style_image = tensor_to_image(style_image)

        cols = st.columns(2)

        titles = ['Content Image', 'Style Image']
        images = [content_image, style_image]

        for i in range(2):
            cols[i].title(titles[i])
            cols[i].image(images[i])

        st.title('Generated Image')
        st.image(generated_image)

else:
    st.warning('Please upload files for the content and style image')