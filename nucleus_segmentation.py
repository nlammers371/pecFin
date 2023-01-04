from ome_zarr.io import parse_url
from ome_zarr.reader import Reader
import napari
import numpy as np

readPath = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/pecFin/HCR_Data/zarr_files/2022_12_15 HCR Hand2 Tbx5a Fgf10a_1.zarr"

# read the image data
store = parse_url(readPath, mode="r").store

reader = Reader(parse_url(readPath))

# nodes may include images, labels etc
nodes = list(reader())

# first node will be the image pixel data
image_node = nodes[0]
image_data = image_node.data

# pull second-smallest image and experiment
im_3 = np.asarray(image_data[2])
# calculate upper resolution limit for display
res_upper = np.percentile(im_3[0,:,:,:],99.999)

viewer = napari.view_image(image_data[2], channel_axis=0, contrast_limits=[0, res_upper])

if __name__ == '__main__':
    napari.run()