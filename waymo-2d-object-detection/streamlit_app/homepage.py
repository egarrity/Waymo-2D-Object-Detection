

import time
import streamlit as st

import numpy as np 
import pandas as pd 
import cv2
"""
#peter's code (-:
def app():
    st.title("Welcome to Object Detection Central")
    st.header("Detecting cars, cyclists and pedestrians 🚗🚴‍♂🚶‍♂️")

    img = cv2.imread('/Users/peterfagan/Code/Waymo-2D-Object-Detection/assets/waymo.jpeg')
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img, width=600)

    page_selection = pd.DataFrame({
      'first column': [1, 2, 3, 4],
      'second column': [10, 20, 30, 40]
    })

    expander = st.beta_expander("FAQ")
    expander.write("Here you could put in some really, really long explanations...")"""
    
    
    
import PIL.Image as Image
from gluoncv import model_zoo, data, utils
from matplotlib import pyplot as plt
import streamlit as st


def vis_image_with_bbox(imgfile):
    net = model_zoo.get_model('center_net_resnet18_v1b_voc', pretrained=True)
    #im_fname = utils.download('https://raw.githubusercontent.com/zhreshold/' +
    #                          'mxnet-ssd/master/data/demo/dog.jpg',
    #                          path='dog.jpg')
    x, img = data.transforms.presets.center_net.load_test("tmpImgFile.jpg", short=512)
    print('Shape of pre-processed image:', x.shape)

    class_IDs, scores, bounding_boxs = net(x)
    #fig = plt.figure()
    ax = utils.viz.plot_bbox(img, bounding_boxs[0], scores[0],
                             class_IDs[0], class_names=net.classes)
    fig = plt.gcf()
    #fig.axes.append(ax)
    plt.draw()
    fn = 'tmpfile.png'
    fig.savefig(fn)

    return fn


def app():
    st.title("Welcome to Object Detection Central")
    st.header("Detecting cars, cyclists and pedestrians 🚗🚴‍♂🚶‍♂️")
    topbox = st.sidebar.selectbox("Choose what to do ", ['Inspect Predictions Visually',
                                                         'Other Options'
                                                         ])

    if topbox == 'Inspect Predictions Visually':
        st.sidebar.subheader('Inspect Predictions')

        uploaded_file = st.file_uploader("Choose an image...", type="jpg")
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image.save('tmpImgFile.jpg')
            st.image(image, caption='Uploaded Image.', use_column_width=True)
            st.write("")
            st.write("Classifying...")
            fn = vis_image_with_bbox(image)
            st.image(fn, use_column_width=True)


if __name__ == '__main__':
    app()

   

