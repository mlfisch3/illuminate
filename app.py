import cv2
import numpy as np
from matplotlib import image as img
import streamlit as st
import time

def run_app():

    @st.cache
    def array_info(array, print_info=True, return_info=False):
        a ={}
        a['dtype'] = array.dtype
        a['ndim'] = array.ndim
        a['shape'] = array.shape
        a['max'] = array.max()
        a['min'] = array.min()
        a['mean'] = array.mean()
        a['std'] = array.std()
        a['size'] = array.size
        a['nonzero'] = np.count_nonzero(array)
        a['layer_variation'] = 0
        
        a['pct'] = 100 * a['nonzero'] / a['size']
        
        if print_info:
            print('{dtype}  {shape}'.format(**a))
            print('nonzero: {nonzero} / {size}  ({pct:.1f} %)'.format(**a))
            print('min:  {min:.2f}   max: {max:.2f}'.format(**a))
            print('mean: {mean:.2f}   std: {std:.2f}'.format(**a), end="")
            if array.ndim > 2:
                a['layer_variation'] = array.std(axis=array.ndim-1).mean()
                print('     layer_variation: {layer_variation:.2f}'.format(**a))

        if return_info:
            return a

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
 
        image_np_info = array_info(image_np, print_info=False, return_info=True)
        image_np_info_str = '\n'.join(sorted({f'{key}: {value}' for key, value in image_np_info.items()}))
        st.sidebar.text("..................................")
        st.sidebar.text("..................................")
        st.sidebar.text("Pixel Statistics [Original Image]:")
        st.sidebar.text("..................................\n")
        st.sidebar.text(image_np_info_str)
        st.sidebar.text("\n..................................")

        st.text(f'Initial Image: {input_file_name}')
        st.image(image_np[:,:,[2,1,0]])

        input_file_name = st.text_input('Download Original Image', input_file_name)
        st.text('Supported file extensions: jpg, jpeg, png, gif, bmp, pdf, svg, eps')
        btn = st.button("Download Original Image")

        if btn:
            with st.spinner(text="Downloading ..."):
                img.imsave(input_file_name, image_np_ai)
                st.text("Download complete")
        
        st.text('------------------------------------------------------------------------------------')
        with st.spinner(text="Processing image ..."):
            start = time.time()
            image_np_ai = adjust_intensity(image_np)
            end = time.time()
            st.text(f'Processing time: {end-start:.5f} s')
        st.text('------------------------------------------------------------------------------------')

        image_np_ai_info = array_info(image_np_ai, print_info=False, return_info=True)
        image_np_ai_info_str = '\n'.join(sorted({f'{key}: {value}' for key, value in image_np_ai_info.items()}))
        st.sidebar.text("\n\n\n\n\n")
        st.sidebar.text("..................................")
        st.sidebar.text("..................................")
        st.sidebar.text("Pixel Statistics [Enhanced Image]:")
        st.sidebar.text("..................................\n")
        st.sidebar.text(image_np_ai_info_str)
        st.sidebar.text("\n..................................")


        processed_file_name = input_file_basename + '_AI' + input_file_ext
        st.text(f'Processed Image: {processed_file_name}')
        st.image(image_np_ai)

        output_file_name = st.text_input('Download Adjusted Intensity Image', processed_file_name)
        st.text('Supported file extensions: jpg, jpeg, png, gif, bmp, pdf, svg, eps')

        button_ai = st.button("Download Enhanced Image", 'adjusted_intensity')
        if button_ai:
            with st.spinner(text="Downloading ..."):
                img.imsave(output_file_name, image_np_ai)
                st.text("Download complete")


if __name__ == '__main__':

    run_app()