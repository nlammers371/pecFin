"""
Display a labels layer above of an image layer using the add_labels and
add_image APIs
"""
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader
import numpy as np
from aicsimageio import AICSImage
from skimage import data
from scipy import ndimage as ndi
from napari_animation import Animation
import napari


# set parameters
filename = "2022_12_21 HCR Prdm1a Robo3 Fgf10a_3"
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


scale_vec = multiscale_attrs[0]["datasets"][0]["coordinateTransformations"][0]["scale"]
channel_names = [channel_metadata[i]["label"] for i in range(len(channel_metadata))]
#colormaps = [channel_metadata[i]["color"] for i in range(len(channel_metadata))]
colormaps = ["red", "blue", "green", "gray"]


viewer = napari.view_image(imData, channel_axis=0, name=channel_names, colormap=colormaps, scale=scale_vec)
viewer.add_labels(label_data[0], name='segmentation', scale=scale_vec)
viewer.window.add_plugin_dock_widget(plugin_name='napari-animation')
napari.run()
# animation = Animation(viewer)
# viewer.update_console({'animation': animation})
# n_steps = 1
# viewer.dims.ndisplay = 3
# viewer.camera.angles = (0.0, 0.0, 90.0)
# animation.capture_keyframe(steps=n_steps)
# viewer.camera.zoom = 2.4
# animation.capture_keyframe(steps=n_steps)
# viewer.camera.angles = (-7.0, 15.7, 62.4)
# animation.capture_keyframe(steps=n_steps)
# viewer.camera.angles = (2.0, -24.4, -36.7)
# animation.capture_keyframe(steps=n_steps)
# viewer.reset_view()
# viewer.camera.angles = (0.0, 0.0, 90.0)
# animation.capture_keyframe(steps=n_steps)
# animation.animate('demo3D.mov', canvas_only=False)