import numpy as np
import napari
import os
import glob2 as glob
import cv2
from aicsimageio import AICSImage
import ntpath
from tqdm import tqdm
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader
from skimage.transform import resize
# from PIL import Image

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

# designate read paths
db_path = "E:\\Nick\\Dropbox (Cole Trapnell's Lab)\\Nick\\pecFin\\HCR_Data\\"
# db_path = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/pecFin/HCR_Data/"
r_seed_list = [443, 242, 253, 903, 437, 371, 883, 910, 503, 921, 108, 730]
suffix_vec = ["_xy", "_zx", "_zy"]
window_size = 768
decon_flag = True
n_samples = 10
overwrite_flag = False
skip_labeled_flag = False
if overwrite_flag:
    skip_labeled_flag = False

# get list of datasets with label priors available for revision
zarr_dir = os.path.join(db_path, 'built_zarr_files', '')
project_list = glob.glob(zarr_dir + '*.zarrlabelpriors')

project_i = 0
print("Generating training slices...")
for project_i in tqdm(range(0, len(project_list))):
# for p in [2]:#range(len(project_list)):
#     wait = 0

    r_seed = r_seed_list[project_i]
    np.random.seed(r_seed)  # set random seed
    filename_raw = path_leaf(project_list[project_i])
    filename = filename_raw.replace(".zarrlabelpriors", "")

    im_read_path = os.path.join(db_path, 'raw', filename[:-2], filename + "_decon.czi")
    # lb_read_path = os.path.join(db_path, 'built_zarr_files', filename + ".zarrlabelpriors")

    # set write path
    write_path = os.path.join(db_path, 'cellpose_training_slices_decon', '')
    write_path_xy = os.path.join(db_path, 'cellpose_training_slices_decon_xy', '')
    if not os.path.isdir(write_path):
        os.makedirs(write_path)
    if not os.path.isdir(write_path_xy):
        os.makedirs(write_path_xy)

    # get list of existing labels (if any)
    # existing_labels = sorted(glob.glob(write_path + "*_seg.npy"))
    # existing_label_names = []
    # for ex in existing_labels:
    #     _, im_name = ntpath.split(ex)
    #     existing_label_names.append(im_name)

    # load in image file
    imObject = AICSImage(im_read_path)
    image_data = np.squeeze(imObject.data)
    image_data = image_data[-1, :, :, :]
    res_raw = imObject.physical_pixel_sizes
    scale_vec = np.asarray(res_raw)

    # load label priors
    # reader_lb = Reader(parse_url(lb_read_path))
    # nodes_lb = list(reader_lb())
    # label_node = nodes_lb[1]
    # label_data = label_node.data[0]

    # randomly choose slices along each direction
    dim_vec = image_data.shape
    xy_slice_indices = np.random.choice(range(dim_vec[0]), n_samples, replace=False)
    xy_id_arr = np.zeros(xy_slice_indices.shape)
    zx_slice_indices = np.random.choice(range(dim_vec[1]), int(n_samples/2), replace=False)
    zx_id_arr = np.ones(zx_slice_indices.shape)
    zy_slice_indices = np.random.choice(range(dim_vec[2]), int(n_samples/2), replace=False)
    zy_id_arr = np.ones(zy_slice_indices.shape)*2

    # combine and shuffle
    slice_num_vec = np.concatenate((xy_slice_indices, zx_slice_indices, zy_slice_indices), axis=0)
    slice_id_vec = np.concatenate((xy_id_arr, zx_id_arr, zy_id_arr), axis=0)
    shuffle_vec = np.random.choice(range(len(slice_id_vec)), len(slice_id_vec), replace=False)
    slice_num_vec = slice_num_vec[shuffle_vec].astype(int)
    slice_id_vec = slice_id_vec[shuffle_vec].astype(int)

    # image_i = 0
    for image_i in range(len(slice_id_vec)):

        # generate save paths for image slice and labels
        slice_id = slice_id_vec[image_i]
        slice_num = slice_num_vec[image_i]
        suffix = suffix_vec[slice_id]

        label_path = write_path + filename + suffix + "_seg.npy"
        slice_path = write_path + filename + suffix + ".tiff"

        rand_prefix = np.random.randint(0, 100000, 1)[0]
        save_name = os.path.join(write_path, f'{rand_prefix:06}' + '_' + filename + suffix + f'{slice_num:04}')

        # skip_flag = True

        # if (label_path not in existing_labels) or overwrite_flag:
        # skip_flag = False
        # print("Starting with raw label priors...")
        if slice_id == 0:
            im_slice = image_data[slice_num, :, :]
            # lb_slice = np.asarray(label_data[slice_num, :, :])

        elif slice_id == 1:
            im_slice = np.squeeze(image_data[:, slice_num, :])
            # lb_slice = np.squeeze(label_data[:, slice_num, :])
            # rescale
            rs_factor = scale_vec[0]/scale_vec[2]
            new_dim = int(rs_factor*dim_vec[0])
            im_slice = resize(im_slice, [new_dim, dim_vec[2]], order=0, anti_aliasing=False)
            # lb_slice = resize(lb_slice, dim_vec[1:], order=0, anti_aliasing=False, preserve_range=True)

        elif slice_id == 2:
            im_slice = np.squeeze(image_data[:, :, slice_num])
            # lb_slice = np.squeeze(label_data[:, :, slice_num])
            # rescale
            rs_factor = scale_vec[0] / scale_vec[1]
            new_dim = int(rs_factor * dim_vec[0])
            im_slice = resize(im_slice, [new_dim, dim_vec[1]], order=0, anti_aliasing=False)
            # lb_slice = resize(lb_slice, dim_vec[1:], order=0, anti_aliasing=False, preserve_range=True)

        # take randon chunk of full slice
        shape_full = np.asarray(im_slice.shape)
        im_lims = shape_full-window_size
        x_start = np.random.choice(range(im_lims[1]), 1)[0].astype(int)
        if im_lims[0] < 0:
            y_start = 0
            im_slice_chunk = im_slice[:, x_start:x_start + window_size]
        else:
            y_start = np.random.choice(range(im_lims[0]), 1)[0].astype(int)
            im_slice_chunk = im_slice[y_start:y_start+window_size, x_start:x_start+window_size]

        # if not skip_flag:
        #     # open viewer
        #     viewer = napari.view_image(im_slice, colormap="gray")
        #     viewer.add_labels(lb_slice, name="Labels")
        #
        #     print('First free label: ' + str(np.max(lb_slice)+1))
        #
        #     napari.run()
        #
        #     # save new  label layer
        #     lb_layer = viewer.layers["Labels"]
        #
        #     # AICSImage(lb_layer.data.astype(np.uint8)).save(lb_path_full)
            # # save
        cv2.imwrite(save_name + ".tiff", im_slice_chunk)
        if slice_id == 0:
            save_name_xy = os.path.join(write_path_xy, f'{rand_prefix:06}' + '_' + filename + suffix + f'{slice_num:04}')
            cv2.imwrite(save_name_xy + ".tiff", im_slice_chunk)
            # np.save(save_name + "_labels.npy", lb_layer.data.astype(np.uint8))
                # time_in_msec = 1000
                # QTimer().singleShot(time_in_msec, app.quit)

    #         # viewer.close()
    #         # cv2.imwrite(label_path + im_name, lb_layer.data)
    #
    #         wait = input("Press Enter to continue to next image. \nPress 'x' then Enter to exit. \nPress 'n' then Enter to move to next experiment.")
    #         if wait == 'x':
    #             break
    #         elif wait == 'n':
    #             break
    #         else:
    #             image_i += 1
    #             print(image_i)
    #
    #     else:
    #         image_i += 1
    #
    # if wait == 'x':
    #     break  # break the loop
    # else:
    #     image_i = 0
    #     project_i += 1



# load label file if it exists
# if os.path.isfile(image_path + lb_name):
#     lb_object = AICSImage(image_path + lb_name)
#     lb_data = np.squeeze(lb_object.data)
#     labels_layer = viewer.add_labels(lb_data, name='annotation')

# if __name__ == '__main__':
#     napari.run()