import numpy as np
import napari
import os
import glob2 as glob
import cv2
from aicsimageio import AICSImage
import ntpath
from skimage.transform import resize
from skimage import io
import cv2
# from PIL import Image

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

# designate read paths
db_path = "E:\\Nick\\Dropbox (Cole Trapnell's Lab)\\Nick\\pecFin\\HCR_Data\\"
training_folder = "cellpose_training_slices_decon2"
ds_factor = 2
new_training_folder = f"cellpose_training_slices_decon2_ds{ds_factor:02}"

if not os.path.isdir(os.path.join(db_path, new_training_folder)):
    os.makedirs(os.path.join(db_path, new_training_folder))

# get list of labels and image slices
image_list = glob.glob(os.path.join(db_path, training_folder, '') + "*.tiff")
label_list = glob.glob(os.path.join(db_path, training_folder, '') + "*_seg.npy")

print("Resizing training slices...")
e_counter = 0
for im in image_list:

    # load
    im_raw = io.imread(im)
    # imObject = AICSImage(im)
    # image_data = np.squeeze(imObject.data)
    if len(im_raw.shape) > 2:
        im_raw = im_raw[:, :, 0]
    if np.max(im_raw) < 1: # some are grayscale
        im_raw = np.round(im_raw * (2 ** 16)).astype(np.uint16)

    if np.max(im_raw) > 1e3:
        im_name = path_leaf(im)
        if ".tiff" in im_name:
            im_name = im_name[:-5]
        else:
            im_name = im_name[:-4]
        shape_curr = np.asarray(im_raw.shape)
        shape_new = np.round(shape_curr/ds_factor).astype(int)
        im_rs = np.round(resize(im_raw, shape_new, order=1, preserve_range=True)).astype(np.uint16)

        # write
        io.imsave((os.path.join(db_path, new_training_folder, im_name + ".tiff")), im_rs)

        # check for label file
        lb_path = os.path.join(db_path, training_folder, im_name + "_seg.npy")

        if os.path.isfile(lb_path):
            lb_raw = np.load(lb_path, allow_pickle=True)
            lb_dict = lb_raw.tolist()
            mask_raw = lb_dict["masks"]

            shape_curr = np.asarray(mask_raw.shape)
            shape_new = np.round(shape_curr / ds_factor).astype(int)
            mask_rs = resize(mask_raw, shape_new, order=0, preserve_range=True).astype(np.uint16)

            lb_dict_new = lb_dict.copy()

            lb_path_out = lb_path = os.path.join(db_path, new_training_folder, im_name + "_seg.npy")
            lb_dict_new["img"] = im_rs
            lb_dict_new["masks"] = mask_rs
            # lb_dict_new["manual_changes"] = []
            lb_dict_new["filename"] = lb_path_out
            # lb_dict_new["modelpath"] = []
            # lb_dict_new.pop("flows", None)
            # lb_dict_new.pop("ismanual", None)
            # lb_dict_new.pop("modelpath", None)
            # lb_dict_new.pop("flowthreshold", None)
            # lb_dict_new.pop("outlines", None)
            # lb_dict_new.pop("manual_changes", None)

            np.save(lb_path_out, np.asarray(lb_dict_new))



