import streamlit as st
import tensorflow as tf
from PIL import Image
import base64

from utils import *

# Streamlit body text
st.title('Neural Style Transfer Demo')
st.write('Find GitHub repository here: https://github.com/charlieblindsay/neural_style_transfer')
st.write('After you have selected a content image and a style image, an image with the \'content\' of the content image and the \'style\' of style image is generated.')

content_images = ['collie.png', 'running_james.jfif', 'running_johny.jpg']
style_images = ['landscape.jpg', 'monet.png', 'scream.jfif']

content_col, style_col = st.columns(2)

content_col.title('Content Images')
style_col.title('Style Images')

for i in range(3):
    content_image = Image.open(f'app_images/{content_images[i]}')
    content_col.image(content_image)

    style_image = Image.open(f'app_images/{style_images[i]}')
    style_col.image(style_image)

content_col.title('Image selection')
style_col.title('----------')
content_image_selected = content_col.selectbox('Please select a content image', options = ['None selected', 1, 2, 3])
style_image_selected = style_col.selectbox('Please select a style image', options = [1, 2, 3])

if content_image_selected != 'None selected':
    with st.spinner('Generating Image. This may take a while...'):
        content_image = load_img_tensor_from_path(f'./app_images/{content_images[content_image_selected - 1]}')
        style_image = load_img_tensor_from_path(f'./app_images/{style_images[style_image_selected - 1]}')

        pretrained_model = load_model('model')
                
        # Passing the content and style image as inputs into the model to return the generated_image_tensor
        generated_image_tensor = pretrained_model(tf.constant(content_image), tf.constant(style_image))[0]
        generated_image = convert_tensor_to_img(generated_image_tensor)

        content_image = convert_tensor_to_img(content_image)
        style_image = convert_tensor_to_img(style_image)

        # Displaying images in Streamlit

        st.title('Content Image')
        st.image(content_image)

        st.title('Style Image')
        st.image(style_image)

        st.title('Generated Image')
        st.image(generated_image)