from ome_zarr.io import parse_url
from ome_zarr.reader import Reader
import napari
import numpy as np
from napari_animation import Animation
from skimage.measure import label, regionprops, regionprops_table
from aicsimageio import AICSImage

# set parameters
filename = "2022_12_21 HCR Prdm1a Robo3 Fgf10a_1"
readPath = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/pecFin/HCR_Data/raw/" + filename[:-2] + "/" + filename + "_decon.czi"
readPathLabels = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/pecFin/HCR_Data/built_zarr_files/" + filename + ".zarrlabels" #"/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/pecFin/HCR_Data/built_zarr_files/" + filename + "labels"
readPathMeta = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/pecFin/HCR_Data/built_zarr_files/" + filename + ".zarr"

#############
# Main image
#############

# read the image data
reader = Reader(parse_url(readPathMeta))

# nodes may include images, labels etc
nodes = list(reader())

# first node will be the image pixel data
image_node = nodes[0]

# load in raw czi file
imObject = AICSImage(readPath)
imData = np.squeeze(imObject.data)

#############
# Labels
#############

# read the image data
reader_lb = Reader(parse_url(readPathLabels))

# nodes may include images, labels etc
nodes_lb = list(reader_lb())

# first node will be the image pixel data
label_node = nodes_lb[1]
label_data = label_node.data

# extract key image attributes
omero_attrs = image_node.root.zarr.root_attrs['omero']
channel_metadata = omero_attrs['channels']  # list of channels and relevant info
multiscale_attrs = image_node.root.zarr.root_attrs['multiscales']
axis_names = multiscale_attrs[0]['axes']
dataset_info = multiscale_attrs[0]['datasets']  # list containing scale factors for each axis

# pull second-smallest image and experiment
#im_3 = np.asarray(image_data[level])
# calculate upper resolution limit for display
#res_upper = np.percentile(im_3[3, :, :, :], 99.999)
# extract useful info
scale_vec = multiscale_attrs[0]["datasets"][0]["coordinateTransformations"][0]["scale"]
channel_names = [channel_metadata[i]["label"] for i in range(len(channel_metadata))]
#colormaps = [channel_metadata[i]["color"] for i in range(len(channel_metadata))]
colormaps = ["red", "blue", "green", "gray"]

viewer = napari.view_image(imData, channel_axis=0, name=channel_names, colormap=colormaps, scale=scale_vec)
labels_layer = viewer.add_labels(label_data[0], name='segmentation', scale=scale_vec)

viewer.theme = "dark"

if __name__ == '__main__':
    napari.run()