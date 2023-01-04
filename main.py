import numpy as np
from aicsimageio import AICSImage
import zarr
import ome_zarr
import napari
from ome_zarr.io import parse_url
from ome_zarr.writer import write_image
import os
import glob2 as glob


# define axis units
units = [None, "um", "um", "um"]

# set paths to raw data
rawPath = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/pecFin/HCR_Data/raw/"
folderName = "2022_12_15 HCR Hand2 Tbx5a Fgf10a"

# set write paths
writeFolder = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/pecFin/HCR_Data"

# get list of valid image files within directory
imageReadDir = rawPath + folderName + '/'
imageList = sorted(glob.glob(imageReadDir + folderName + "_?.czi"))

# make write directory
if os.path.isdir(writeFolder) != True:
   os.makedirs(writeFolder)

# iterate through list
im = 0

# load image
print("Loading image...")
imRaw = AICSImage(imageList[im])

# convert to zarr
print("Squeezing...")
imData = imRaw.data
imData = np.squeeze(imData)

print("Converting to zarr format")
imZarr = zarr.zeros(imData.shape)
imZarr[:, :, :] = imData

axis_names = tuple("czyx")
#scaler = ome_zarr.scale.Scaler()
#mip = scaler.local_mean(imZarr)

# determine chunk size
size_xy = 128
size_z = 3

# define attribute dictionary
res_raw = imRaw.physical_pixel_sizes
res_array = np.asarray(res_raw)
res_array = np.insert(res_array, 0, 1)

att_dict = {
    "transform": {
        "axes": [axis_names],
        "scale": res_array,
        "translate": [
            0.0,
            0.0,
            0.0
        ],
        "units": units
    }
}

# make write directory
imFolder = writeFolder + '/zarr_files/'
imName = folderName + '_' + str(im+1) + '.zarr'
if os.path.isdir(imFolder) != True:
   os.makedirs(imFolder)

# write the image data
store = parse_url(imFolder + imName, mode="w").store
root = zarr.group(store=store)
write_image(image=imZarr, group=root, axes="czyx", storage_options=dict(chunks=(1, size_z, size_xy, size_xy)))
root.zattrs = att_dict


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
   print("running...")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
