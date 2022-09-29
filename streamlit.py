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

if content_image is not None and style_image is not None:
    # bytes_data = uploaded_file.read()
    # st.write("filename:", uploaded_file.name)
    # st.write(bytes_data)

    # dataframe = pd.read_csv(uploaded_file)
    # st.dataframe(data=dataframe)

    # file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type}
    # st.write(file_details)
    # img = load_image(uploaded_file)
    # st.image(img)

    # content_path = tf.keras.utils.get_file('YellowLabradorLooking_new.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')

    save_uploaded_files([{'type': 'content', 'img': content_image}, {'type': 'style', 'img': style_image}])

    content_image = load_img('./content.png')
    style_image = load_img('./style.png')

    hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
    stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]
    generated_image = tensor_to_image(stylized_image)
    st.image(generated_image)
