import streamlit as st
import tensorflow as tf

import os
# Load compressed models from tensorflow_hub
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'

from utils import *

def save_uploaded_files(uploaded_files):
    for file in uploaded_files:
        file_name = file['type']
        file_name_extension = f'{file_name}.png'
        with open(file_name_extension,"wb") as f:
            f.write(file['img'].getbuffer())

form = st.form(key='my_form')

content_image = form.file_uploader("Choose a content image")
style_image = form.file_uploader("Choose a style image")

possible_content_layers = ['block1_conv1', 'block1_conv2', 'block2_conv1', 'block2_conv2', 'block3_conv1', 'block3_conv2', 'block3_conv3', 'block3_conv4', 'block4_conv1', 'block4_conv2', 'block4_conv3', 'block4_conv4', 'block5_conv1', 'block5_conv2', 'block5_conv3', 'block5_conv4']
content_layer_input = form.selectbox(label='A content layer is chosen to see how similar the \'content\' of the content and generated images are. Choose how deep into the CNN you want the content layer to be:', options=possible_content_layers, index=5)

style_weight_input = form.slider(label='Choose how closely you want the style of the generated image to match that of the style image:', value=3, step=1, min_value=0, max_value=5)

submit_button = form.form_submit_button(label='Submit')

if content_image is not None and style_image is not None:
    with st.spinner('Generating Image. This make take a while...'):
        style_weight= float(f'1e{style_weight_input - 5}')
        content_weight=1e3

        save_uploaded_files([{'type': 'content', 'img': content_image}, {'type': 'style', 'img': style_image}])

        content_image = load_img('./content.png')
        style_image = load_img('./style.png')

        content_layers = [content_layer_input]

        style_layers = ['block1_conv1',
                        'block2_conv1',
                        'block3_conv1', 
                        'block4_conv1', 
                        'block5_conv1']

        num_content_layers = len(content_layers)
        num_style_layers = len(style_layers)

        def load_model():
            vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
            vgg.trainable = False
            return vgg
        
        def vgg_layers(layer_names):
            """ Creates a VGG model that returns a list of intermediate output values."""
            # Load our model. Load pretrained VGG, trained on ImageNet data
            vgg = load_model()

            outputs = [vgg.get_layer(name).output for name in layer_names]

            model = tf.keras.Model([vgg.input], outputs)
            return model

        def gram_matrix(input_tensor):
            result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
            input_shape = tf.shape(input_tensor)
            num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
            return result/(num_locations)

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

        extractor = StyleContentModel(style_layers, content_layers)

        results = extractor(tf.constant(content_image))

        style_targets = extractor(style_image)['style']
        content_targets = extractor(content_image)['content']

        image = tf.Variable(content_image)

        def clip_0_1(image):
            return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

        opt = tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)


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
        
        epochs = 10
        steps_per_epoch = 3

        step = 0
        for n in range(epochs):
            for m in range(steps_per_epoch):
                step += 1
                train_step(image)
        
        generated_image = tensor_to_image(image)

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