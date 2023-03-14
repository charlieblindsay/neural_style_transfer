""" Streamlit app which uses functions from utils.py

To run the streamlit app locally, enter this command in the terminal: 
streamlit run c:/PATH/TO/CURRENT/DIRECTORY/app.py
"""
import streamlit as st
import tensorflow as tf

from utils import *

content_layers = ['block5_conv1'] 
# content_layers = ['block2_conv1']

style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1']

style_weight=1e4
content_weight=1e-2

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

style_extractor = vgg_layers(style_layers)


@st.cache_resource # saves the output
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

def style_content_loss(outputs):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) 
                           for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) 
                             for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers
    loss = style_loss + content_loss
    return loss

@tf.function()
def train_step(image):
  with tf.GradientTape() as tape:
    outputs = extractor(image)
    loss = style_content_loss(outputs)

  grad = tape.gradient(loss, image)
  opt.apply_gradients([(grad, image)])
  image.assign(clip_0_1(image))


# Streamlit body text
st.title('Neural Style Transfer Demo')
st.write('Find GitHub repository here: https://github.com/charlieblindsay/neural_style_transfer')
st.write('After the user has uploaded a content image and a style image, an image with the \'content\' of the content image and the \'style\' of style image is generated.')
st.subheader('File upload')

# Streamlit file uploader widgets
content_image = st.file_uploader("Choose a content image. This should be a photo.")
style_image = st.file_uploader('Choose a style image. This should be a piece of art, e.g. a painting')

opt = tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)   

# Once both the content and style image have been uploaded by the user, the if statement is entered
if content_image is not None and style_image is not None:
    with st.spinner('Generating Image. This may take a while...'):

        save_uploaded_files([{'type': 'content', 'img': content_image}, {'type': 'style', 'img': style_image}])

        content_image = load_img_tensor_from_path('./content.png')
        style_image = load_img_tensor_from_path('./style.png')

        extractor = StyleContentModel(style_layers, content_layers)
        style_targets = extractor(style_image)['style']
        content_targets = extractor(content_image)['content']

        image = tf.Variable(content_image)

        epochs = 1
        steps_per_epoch = 100

        step = 0
        for n in range(epochs):
            for m in range(steps_per_epoch):
                step += 1
                train_step(image)

        img = convert_tensor_to_img(image)

        # pretrained_model = load_model('model')
        
        # Passing the content and style image as inputs into the model to return the generated_image_tensor
        # generated_image_tensor = pretrained_model(tf.constant(content_image), tf.constant(style_image))[0]
        # generated_image = convert_tensor_to_img(generated_image_tensor)

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
        st.image(img)

# Whilst the user has not uploaded the images, this message is displayed:
else:
    st.warning('Please upload files for the content and style image')