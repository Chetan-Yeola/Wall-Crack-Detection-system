import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import torchvision
import numpy as np
import os
import cv2

cwd = os.getcwd()
from PIL import Image
import time
import copy
import random
import streamlit as st
import re
import shutil
import matplotlib.pyplot as plt
import predict
import matplotlib.image as mpimg

upload = st.file_uploader(" Choose a wall image", type=None)

if upload is not None:
    # name = upload.name
    # ext = name.split('.')
    # if ext[1]=='pdf':
    #     img_p= pdf2im(upload)
    #     image= Image.open(img_p)
    #     img= np.asarray(image)
    # else:
    image = Image.open(upload)
    bytes_data = upload.getvalue()

    img = np.asarray(image)

    st.image(image, caption='Uploaded image', use_column_width=True)
    output_image = predict.predict_on_crops(img, 128, 128)
    st.image(output_image)
    # plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
