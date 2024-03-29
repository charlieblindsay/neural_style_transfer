""" Streamlit app which uses functions from utils.py

To run the streamlit app locally, enter this command in the terminal: 
streamlit run c:/PATH/TO/CURRENT/DIRECTORY/app.py
"""
import streamlit as st
import tensorflow as tf

from utils import *

def save_uploaded_files(uploaded_files: list[dict]):
    """Writes the images uploaded by the user to the current working directory

    Args:
        uploaded_files (list[dict]): List of all files to save. 
        Each file is represented by a dict with parameters:
        - 'type' (str): The type of image being uploaded (i.e. either 'content' or 'style')
        - 'img' (st.file_uploader.UploadedFile): The image file uploaded by the user
    """
    for file in uploaded_files:
        file_name = file['type']
        file_name_with_extension = f'{file_name}.png'
        with open(file_name_with_extension,"wb") as f:
            f.write(file['img'].getbuffer())

# Streamlit body text
st.title('Neural Style Transfer Demo')
st.write('Find GitHub repository here: https://github.com/charlieblindsay/neural_style_transfer')
st.write('After you have uploaded a content image and a style image, an image with the \'content\' of the content image and the \'style\' of style image is generated.')
st.subheader('File upload')

# Streamlit file uploader widgets
content_image = st.file_uploader("Choose a content image. This should be a photo.")
style_image = st.file_uploader('Choose a style image. This should be a piece of art, e.g. a painting')

# Once both the content and style image have been uploaded by the user, the if statement is entered
if content_image is not None and style_image is not None:
    with st.spinner('Generating Image. This may take a while...'):
        save_uploaded_files([{'type': 'content', 'img': content_image}, {'type': 'style', 'img': style_image}])

        content_image = load_img_tensor_from_path('./content.png')
        style_image = load_img_tensor_from_path('./style.png')

        pretrained_model = load_model('model')
        
        # Passing the content and style image as inputs into the model to return the generated_image_tensor
        generated_image_tensor = pretrained_model(tf.constant(content_image), tf.constant(style_image))[0]
        generated_image = convert_tensor_to_img(generated_image_tensor)

        content_image = convert_tensor_to_img(content_image)
        style_image = convert_tensor_to_img(style_image)

        # Displaying images in Streamlit
        cols = st.columns(2)

        titles = ['Content Image', 'Style Image']
        images = [content_image, style_image]

        for i in range(2):
            cols[i].title(titles[i])
            cols[i].image(images[i])

        st.title('Generated Image')
        st.image(generated_image)

# Whilst the user has not uploaded the images, this message is displayed:
else:
    st.warning('Please upload files for the content and style image')