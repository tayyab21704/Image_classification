import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import  load_model
import streamlit as st
import numpy as np 
from PIL import Image

st.header('Image Classification Model')
model = load_model('C:\\cnn_project\\Image_classify.h5')
data_cat = ['apple',
 'banana',
 'beetroot',
 'bell pepper',
 'cabbage',
 'capsicum',
 'carrot',
 'cauliflower',
 'chilli pepper',
 'corn',
 'cucumber',
 'eggplant',
 'garlic',
 'ginger',
 'grapes',
 'jalepeno',
 'kiwi',
 'lemon',
 'lettuce',
 'ango',
 'onion',
 'orange',
 'paprika',
 'pear',
 'peas',
 'pineapple',
 'pomegranate',
 'potato',
 'raddish',
 'oy beans',
 'pinach',
 'weetcorn',
 'weetpotato',
 'tomato',
 'turnip',
 'watermelon']

img_height = 180
img_width = 180

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_arr = tf.keras.utils.img_to_array(image)
    img_arr = tf.image.resize(img_arr, (img_height, img_width))
    img_bat = tf.expand_dims(img_arr, 0)

    predict = model.predict(img_bat)
    score = tf.nn.softmax(predict)
    st.image(image, width=200)
    st.write('Veg/Fruit in image is '+ data_cat[np.argmax(score)])
    st.write('With accuracy of '+ str(np.max(score)*100))
else:
    st.write("Please upload an image!")