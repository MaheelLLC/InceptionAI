import cv2
import numpy as np
import os

def resize_and_pad(img, size=(512,512), pad_color=0):
    h, w = img.shape[:2]

    # Checks the case when the image is already fine
    if h == 512 and w == 512:
        return img
    
    sh, sw = size
    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv2.INTER_AREA
    else: # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = w/h

    # compute scaling and pad sizing
    if aspect > 1: # horizontal image
        new_w = sw
        new_h = np.round(new_w / aspect).astype(int)
        pad_vert = (sh-new_h)/2
        pad_top = np.floor(pad_vert).astype(int)
        pad_bot = np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1: # vertical image
        new_h = sh
        new_w = np.round(new_h * aspect).astype(int)
        pad_horz = (sw-new_w)/2
        pad_left = np.floor(pad_horz).astype(int)
        pad_right = np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else: # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # set pad color
    # color image but only one color provided
    if len(img.shape) == 3 and not isinstance(pad_color, 
                                             (list, tuple, np.ndarray)):
        pad_color = [pad_color]*3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left,
                                    pad_right, cv2.BORDER_CONSTANT, 
                                    value = pad_color)
    return scaled_img


# Directory containing images
input_dir = "/home/maheel/Documents/Python/Route_2/New_Tumor/Testing/notumor"
output_dir = "/home/maheel/Documents/Python/Route_2/New_Tumor/Testing/asp_notumor"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Process each image
for filename in os.listdir(input_dir):
    if filename.endswith(".jpg"):
        img = cv2.imread(os.path.join(input_dir, filename))
        resized_img = resize_and_pad(img)
        cv2.imwrite(os.path.join(output_dir, filename), resized_img)
