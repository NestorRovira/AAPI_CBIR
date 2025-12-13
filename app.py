import time
import torch
import faiss
import pathlib
from PIL import Image
import numpy as np
import pandas as pd
import os
import time

import streamlit as st
from streamlit_cropper import st_cropper


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

st.set_page_config(layout="wide")

device = torch.device('cpu')

FILES_PATH = str(pathlib.Path().resolve())

# Path in which the images should be located
IMAGES_PATH = os.path.join(FILES_PATH, 'images')
# Path in which the database should be located
DB_PATH = os.path.join(FILES_PATH, 'database')

DB_FILE = 'db.csv' # name of the database

def get_image_list():
    df = pd.read_csv(os.path.join(DB_PATH, DB_FILE))
    image_list = list(df.image.values)
    return image_list

def retrieve_image(img_query, feature_extractor, n_imgs=11):
    if (feature_extractor == 'Extractor 1'):
        # TODO: select the database according to the feature extractor
        # Function to preprocess and extract features
        model_feature_extractor = ...
        indexer = faiss.read_index(os.path.join(DB_PATH,  'feat_extract_1.index'))
    elif (feature_extractor == 'Extractor 2'):
        model_feature_extractor = ...
        indexer = faiss.read_index(os.path.join(DB_PATH,  'feat_extract_2.index'))

    # TODO: Modify accordingly
    embeddings = model_feature_extractor(img_query)
    vector = np.float32(embeddings)
    faiss.normalize_L2(vector)

    _, indices = indexer.search(vector, k=n_imgs)

    return indices[0]

def main():
    st.title('CBIR IMAGE SEARCH')
    
    col1, col2 = st.columns(2)

    with col1:
        st.header('QUERY')

        st.subheader('Choose feature extractor')
        # TODO: Adapt to the type of feature extraction methods used.
        option = st.selectbox('.', ('Extractor 1', 'Extractor 2'))

        st.subheader('Upload image')
        img_file = st.file_uploader(label='.', type=['png', 'jpg'])

        if img_file:
            img = Image.open(img_file)
            # Get a cropped image from the frontend
            cropped_img = st_cropper(img, realtime_update=True, box_color='#FF0004')
            
            # Manipulate cropped image at will
            st.write("Preview")
            _ = cropped_img.thumbnail((150,150))
            st.image(cropped_img)

    with col2:
        st.header('RESULT')
        if img_file:
            st.markdown('**Retrieving .......**')
            start = time.time()

            retriev = retrieve_image(cropped_img, option, n_imgs=11)
            image_list = get_image_list()

            end = time.time()
            st.markdown('**Finish in ' + str(end - start) + ' seconds**')

            col3, col4 = st.columns(2)

            with col3:
                image = Image.open(os.path.join(IMAGES_PATH, image_list[retriev[0]]))
                st.image(image, use_column_width = 'always')

            with col4:
                image = Image.open(os.path.join(IMAGES_PATH, image_list[retriev[1]]))
                st.image(image, use_column_width = 'always')

            col5, col6, col7 = st.columns(3)

            with col5:
                for u in range(2, 11, 3):
                    image = Image.open(os.path.join(IMAGES_PATH, image_list[retriev[u]]))
                    st.image(image, use_column_width = 'always')

            with col6:
                for u in range(3, 11, 3):
                    image = Image.open(os.path.join(IMAGES_PATH, image_list[retriev[u]]))
                    st.image(image, use_column_width = 'always')

            with col7:
                for u in range(4, 11, 3):
                    image = Image.open(os.path.join(IMAGES_PATH, image_list[retriev[u]]))
                    st.image(image, use_column_width = 'always')

if __name__ == '__main__':
    main()