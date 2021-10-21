import cv2
import numpy as np
from matplotlib import image as img
import streamlit as st
from PIL import Image

def run_app():

    @st.cache
    def adjust_intensity(array):
        return cv2.intensity_transform.BIMEF(array[:,:,[2,1,0]])

    fImage = st.sidebar.file_uploader("Upload image file:")

    if fImage is not None:

        input_file_name = str(fImage.__dict__['name'])
        input_file_ext = '.' + str(input_file_name.split('.')[-1])
        input_file_basename = input_file_name.replace(input_file_ext, '')
        np_array = np.frombuffer(fImage.getvalue(), np.uint8) 
        image_np = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
  
        st.text(f'Initial Image: {input_file_name}')
        st.image(image_np[:,:,[2,1,0]])

        input_file_name = st.text_input('Download Original Image', input_file_name)
        st.text('Supported file extensions: jpg, jpeg, png, gif, bmp, pdf, svg, eps')
        btn = st.button("Download")

        if btn:
            with st.spinner(text="Downloading ..."):
                img.imsave(input_file_name, image_np_ai)
                st.text("Download complete")

        with st.spinner(text="Processing image ..."):
            prog_bar = st.progress(0)
            image_np_ai = adjust_intensity(image_np)

        processed_file_name = input_file_basename + '_AI' + input_file_ext
        st.text(f'Processed Image: {processed_file_name}')
        st.image(image_np_ai)

        output_file_name = st.text_input('Download Adjusted Intensity Image', processed_file_name)
        st.text('Supported file extensions: jpg, jpeg, png, gif, bmp, pdf, svg, eps')

        button_ai = st.button("Download", 'adjusted_intensity')
        if button_ai:
            with st.spinner(text="Downloading ..."):
                img.imsave(output_file_name, image_np_ai)
                st.text("Download complete")


if __name__ == '__main__':

    run_app()