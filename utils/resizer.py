# This file converts .jpeg images to .jpg
# Wondering if this is why I am having a problem using the tattoo dataset that is .jpeg so I'm converting to .jpg and then trying that.

# importing the module
from PIL import Image
import os
import imutils
from imutils import paths
import cv2

outdir = "/Users/peppermint/Desktop/multi-modal-jina/utils/results_resize"
imagePaths = list(paths.list_images("/Users/peppermint/Desktop/multi-modal-jina/utils/results"))

for image in imagePaths:
    print(image.split("/")[-1])
    outpath = image.split("/")[-1]
    outpath = outpath.split(".")[0] + ".jpg"
    im = Image.open(image)
    rgb_im = im.convert("RGB")
    save_path = outdir + outpath
    print(save_path)
    rgb_im.save(save_path) 

