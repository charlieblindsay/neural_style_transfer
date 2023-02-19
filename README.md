# Neural Style Transfer

Streamlit app: https://charlieblindsay-neural-style-transfer-app-5cov8f.streamlit.app/

![](https://github.com/charlieblindsay/neural_style_transfer/blob/main/screen%20recording.gif)

![alt text](nst_app.jpg)

After the user has uploaded a content image and a style image, an image with the 'content' of the content image and the 'style' of style image is generated.

### How is this done?
Machine Learning!

The generated image is initialized as the content image, on each iteration of an optimization algorithm (e.g. gradient descent), the image's pixel content is changed to reduce a cost function.

This cost function is the scaled adding of 2 components: the content cost and the style cost.
- The content cost function takes the content image and generated image as input. It measures the Euclidean distance between the activations of the content and generated image at a chosen layer in a (pre-trained) convolutional neural network (CNN). 
- The style cost function takes the style image and generated image as input. Firstly, the style matrices of the style and generated image are calculated for each layer in the CNN; the style matrix is an n_c x n_c (where n_c is the number of 'channels', a.k.a. features) matrix which represents the correlation of different features with eachother. The style cost is the Frobenius norm of the difference between these 2 matrices, averaged across all layers in the CNN.

##### Aside
I think this mathematical defition of the 'style' of an image is intriguing; 2 images have a similar style if the same low-level features, e.g. lines and curves, and high-level features occur together.

### Technical Note
VGG-19 was used (with the weights pre-trained on imagenet data).
