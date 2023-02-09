import plotly.express as px
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import glob2 as glob
from skimage.measure import label, regionprops, regionprops_table
import math


def ellipsoid_axis_lengths(central_moments):
    """Compute ellipsoid major, intermediate and minor axis length.

    Parameters
    ----------
    central_moments : ndarray
        Array of central moments as given by ``moments_central`` with order 2.

    Returns
    -------
    axis_lengths: tuple of float
        The ellipsoid axis lengths in descending order.
    """
    m0 = central_moments[0, 0, 0]
    sxx = central_moments[2, 0, 0] / m0
    syy = central_moments[0, 2, 0] / m0
    szz = central_moments[0, 0, 2] / m0
    sxy = central_moments[1, 1, 0] / m0
    sxz = central_moments[1, 0, 1] / m0
    syz = central_moments[0, 1, 1] / m0
    S = np.asarray([[sxx, sxy, sxz], [sxy, syy, syz], [sxz, syz, szz]])
    # determine eigenvalues in descending order
    eigvals, eigvecs = np.linalg.eig(S)
    si = np.argsort(eigvals)
    si = si[::-1]
    eigvals = eigvals[si]
    eigvecs[:, si] = eigvecs[:, si]

    return tuple([math.sqrt(20.0 * e) for e in eigvals]), eigvecs

def extract_nucleus_stats(dataRoot, filename, level):

    # get list of datasets with nucleus labels
    image_list = sorted(glob.glob(dataRoot + "*.zarrlabels"))

    for im in range(len(image_list)):
        labelPath = image_list[im]
        labelName = image_list[im].replace(dataRoot, '', 1)
        labelName = labelName.replace('.zarrlabels', '')
        imagePath = dataRoot + labelName + ".zarr"
        savePath = dataRoot + labelName + '_nucleus_props.csv'

        reader = Reader(parse_url(imagePath))
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

        df.to_csv(savePath)

if __name__ == "__main__":
    # define some variables
    level = 1
    filename = "2022_12_15 HCR Hand2 Tbx5a Fgf10a_1"
    dataRoot = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/pecFin/HCR_Data/built_zarr_files/"
    labelRoot = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/pecFin/HCR_Data/built_zarr_files/"

    extract_nucleus_stats(dataRoot, filename, level)