import plotly.express as px
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import json
from skimage.measure import label, regionprops, regionprops_table
from pecfin_segmentation import ellipsoid_axis_lengths

# define some variables
level = 1
filename = "2022_12_15 HCR Sox9a Myod1 Col11a2_2"
dataRoot = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/pecFin/HCR_Data/built_zarr_files/"
labelRoot = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/pecFin/HCR_Data/built_zarr_files/"

dataPath = dataRoot + filename + ".zarr"
labelPath = labelRoot + filename + ".zarrlabels"
propPath = dataRoot + filename + '_nucleus_props.csv'

reader = Reader(parse_url(dataPath))
#
# # nodes may include images, labels etc
nodes = list(reader())
#
# # first node will be the image pixel data
image_node = nodes[0]
image_data = image_node.data

#############
# Labels
#############

# read the image data
# store_lb = parse_url(labelPath, mode="r").store
reader_lb = Reader(parse_url(labelPath))

# nodes may include images, labels etc
nodes_lb = list(reader_lb())

# first node will be the image pixel data
label_node = nodes_lb[1]
label_data = label_node.data

# extract key image attributes
# omero_attrs = image_node.root.zarr.root_attrs['omero']
# channel_metadata = omero_attrs['channels']  # list of channels and relevant info
multiscale_attrs = image_node.root.zarr.root_attrs['multiscales']

# extract useful info
scale_vec = multiscale_attrs[0]["datasets"][level]["coordinateTransformations"][0]["scale"]

# add layer of mask centroids
label_array = np.asarray(label_data[level].compute())
regions = regionprops(label_array)

centroid_array = np.empty((len(regions), 3))
for rgi, rg in enumerate(regions):
    centroid_array[rgi, :] = np.multiply(rg.centroid, scale_vec)

# convert centroid array to data frame
df = pd.DataFrame(centroid_array, columns=["Z", "Y", "X"])

# add additional info
area_vec = []
for rgi, rg in enumerate(regions):
    area_vec.append(rg.area)
df = df.assign(Area=np.asarray(area_vec))

# calculate axis lengths
axis_array = np.empty((len(regions), 3))
vec_list = []
for rgi, rg in enumerate(regions):
    moments = rg['moments_central']
    axes, axis_dirs = ellipsoid_axis_lengths(moments)
    axis_array[rgi, :] = np.multiply(axes, scale_vec)
    vec_list.append(axis_dirs)

df2 = pd.DataFrame(axis_array, columns=["Axis_1", "Axis_2", "Axis_3"])
df2 = df2.assign(axis_dirs=vec_list)
df = pd.concat([df, df2], axis=1)