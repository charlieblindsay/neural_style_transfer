import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub

from utils import *

def save_uploaded_files(uploaded_files):
    for file in uploaded_files:
        file_name = file['type']
        file_name_extension = f'{file_name}.png'
        with open(file_name_extension,"wb") as f:
            f.write(file['img'].getbuffer())
    st.success("Saved Files")

content_image = st.file_uploader("Choose a content image")
style_image = st.file_uploader("Choose a style image")

@st.cache
def load_model():
    return hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

if content_image is not None and style_image is not None:

    save_uploaded_files([{'type': 'content', 'img': content_image}, {'type': 'style', 'img': style_image}])

    content_image = load_img('./content.png')
    style_image = load_img('./style.png')

    hub_model = load_model()
    stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]
    generated_image = tensor_to_image(stylized_image)
    st.image(generated_image)
