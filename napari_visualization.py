from ome_zarr.io import parse_url
from ome_zarr.reader import Reader
import napari
import numpy as np

readPath = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/pecFin/HCR_Data/built_zarr_files/2022_12_22 HCR Sox9a Tbx5a Emilin3a_1.zarr"
# "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/pecFin/HCR_Data/zarr_files/2022_12_15 HCR Hand2 Tbx5a Fgf10a_1.zarr"

# read the image data
store = parse_url(readPath, mode="r").store

reader = Reader(parse_url(readPath))

# nodes may include images, labels etc
nodes = list(reader())

# first node will be the image pixel data
image_node = nodes[0]
image_data = image_node.data

# extract key image attributes
omero_attrs = image_node.root.zarr.root_attrs['omero']
channel_metadata = omero_attrs['channels']  # list of channels and relevant info
multiscale_attrs = image_node.root.zarr.root_attrs['multiscales']
axis_names = multiscale_attrs[0]['axes']
dataset_info = multiscale_attrs[0]['datasets']  # list containing scale factors for each axis

# pull second-smallest image and experiment
im_3 = np.asarray(image_data[0])
# calculate upper resolution limit for display
res_upper = np.percentile(im_3[0, :, :, :], 99.999)

viewer = napari.view_image(image_data[0], channel_axis=0, contrast_limits=[0, res_upper])

if __name__ == '__main__':
    napari.run()