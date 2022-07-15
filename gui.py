import streamlit as st
import imageio as iio
import model
import pandas as pd

# Layout of the app
st.header("Predict Retinal layers from OCT image")
st.write("Choose any image and get corresponding binary mask:")
img_stk = st.file_uploader("Upload image stack...")
if(img_stk != None):
    img_stk = iio.mimread(img_stk)
    if st.checkbox('Enable Training'):
        mask_stk = st.file_uploader("Upload corresponding mask stack...")
        if(mask_stk != None):
            mask_stk = iio.mimread(mask_stk)
            if st.button('Train'):
                score, loss, pred = model.train(img_stk, mask_stk)
                iio.mimwrite("pred/predictions.tiff", pred)
                score_data = pd.DataFrame(score, columns=["Train IOU score", "Validation IOU score"])
                st.line_chart(score_data)

                loss_data = pd.DataFrame(loss, columns=["Train loss", "Validation loss"])
                st.line_chart(loss_data)
    else:
        if st.button('Predict'):
            pred = model.test(img_stk)
            iio.mimwrite("pred/predictions.tiff", pred)
            