import plotly.express as px
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import glob2 as glob
from skimage.measure import label, regionprops, regionprops_table
import math
import os.path

def plot_fin_ellipsoids():
    import math
    import plotly.graph_objects as go
    from plotly.offline import iplot, init_notebook_mode
    from plotly.graph_objs import Mesh3d
    import numpy as np

    axis_lengths = np.array([11.89184616, 4.82438505, 1.49761732])

    a = axis_lengths[2]
    b = axis_lengths[1]
    c = axis_lengths[0]

    pi = math.pi
    phi = np.linspace(0, 2 * pi)
    theta = np.linspace(-pi / 2, pi / 2)
    phi, theta = np.meshgrid(phi, theta)
    x = np.cos(theta) * np.sin(phi) * a
    y = np.cos(theta) * np.cos(phi) * b
    z = np.sin(theta) * c

    # define a transformation matrix
    R = np.array([[-0.23933507, -0.90749994, 0.34519933],
                  [0.15574894, 0.31504453, 0.93621003],
                  [0.95836371, -0.27783232, -0.06594095]])
    T = np.zeros((4, 4))
    C = np.array([4.47448559, 192.68796498, 138.36717904])
    T[0:3, 0:3] = R
    T[3, 3] = 1
    T[0:3, 3] = C

    x = x.flatten()
    y = y.flatten()
    z = z.flatten()

    xtf = []
    ytf = []
    ztf = []

    for i in range(len(x)):
        xyz = np.array([z[i], y[i], x[i], 1]).reshape(4, 1)
        xyz_tf = np.matmul(T, xyz)
        xyz_tf.flatten()
        xtf.append(xyz_tf[2][0])
        ytf.append(xyz_tf[1][0])
        ztf.append(xyz_tf[0][0])

    fig = go.Figure(data=[go.Mesh3d(x=xtf,
                                    y=ytf,
                                    z=ztf,
                                    alphahull=0)])

    # fig.update_layout(
    #     scene = dict(
    #         xaxis = dict(nticks=4, range=[-30,30]+C[0],),
    #                      yaxis = dict(nticks=4, range=[-30,30]+C[1],),
    #                      zaxis = dict(nticks=4, range=[-30,30]+C[2],),))

    fig.show()
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

def extract_nucleus_stats(dataRoot, level, mRNA_channels=None, resetClassFlag=False):

    # set default
    if mRNA_channels is None:
        mRNA_channels = [0, 1, 2]

    # get list of datasets with nucleus labels
    image_list = sorted(glob.glob(dataRoot + "*.zarrlabels"))

    for im in range(len(image_list)):

        labelPath = image_list[im]
        labelName = image_list[im].replace(dataRoot, '', 1)
        labelName = labelName.replace('.zarrlabels', '')
        imagePath = dataRoot + labelName + ".zarr"
        savePath = dataRoot + labelName + '_nucleus_props.csv'

        resetClassFlag = (not os.path.isfile(savePath)) | resetClassFlag

        print('Extracting stats for ' + labelName)
        reader = Reader(parse_url(imagePath))
        #
        # # nodes may include images, labels etc
        nodes = list(reader())
        #
        # # first node will be the image pixel data
        image_node = nodes[0]
        #image_data = image_node.data

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
        nucleus_df = pd.DataFrame(centroid_array, columns=["Z", "Y", "X"])

        # add additional info
        area_vec = []
        for rgi, rg in enumerate(regions):
            area_vec.append(rg.area)
        nuclues_df = nucleus_df.assign(Area=np.asarray(area_vec))

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
            nucleus_df = pd.concat([nucleus_df, df2], axis=1)

            nucleus_df.to_csv(savePath)

        if not resetClassFlag:
            # check for class column
            # load nucleus centroid data frame
            nucleus_df_orig = pd.read_csv(savePath, index_col=0)

            if 'pecc_fin_flag' in nucleus_df_orig.columns:
                if nucleus_df.shape[0] == nucleus_df_orig.shape[0]:
                    nucleus_df['pec_fin_flag'] = nucleus_df_orig['pec_fin_flag']
                else:
                    raise Warning("Attempted to extract old pec fin assignments, but old and new datasets have different sizes. Skipping.")
            else:
                raise Warning(
                    "Attempted to extract old pec fin assignments, but 'pec_fin_flag' field does not exist. Skipping.")
            # find and remive extraneous columns from data frame
            colnames = nucleus_df.columns

            clean_indices = [i for i, item in enumerate(colnames) if "Unnamed" not in item]
            nucleus_df = nucleus_df.iloc[:, clean_indices]

        ################################
        # Calculate mRNA levels in each nucleus

        image_array = image_data[level]

        # compute each channel of image array separately to avoid dask error
        im_array_list = []
        for ch in mRNA_channels:
            im_array_list.append(np.asarray(image_array[ch, :, :, :].compute()))

        # initialize empty columns
        for ch in channel_names:
            nucleus_df[ch] = np.nan
            nucleus_df[ch + "_mean"] = np.nan

        # calculate mRNA levels
        for rgi, rg in enumerate(regions):

            # iterate through channels
            nc_coords = rg.coords.astype(int)
            n_pix = nc_coords.shape[0]
            for chi, im_ch in enumerate(im_array_list):
                # nc_ch_coords = np.concatenate((np.ones((n_pix,1))*ch, nc_coords), axis=1).astype(int)
                #im_ch = im_array_list[ch]
                mRNA_integral = np.sum(im_ch[tuple(nc_coords.T)])

                nucleus_df[channel_names[chi]].iloc[rgi] = mRNA_integral
                nucleus_df[channel_names[chi] + "_mean"].iloc[rgi] = mRNA_integral / n_pix

        # normalize
        for ch in channel_names:
            int_max = np.max(nucleus_df[ch])
            mean_max = np.max(nucleus_df[ch + "_mean"])
            nucleus_df[ch] = nucleus_df[ch] / int_max
            nucleus_df[ch + "_mean"] = nucleus_df[ch + "_mean"] / mean_max


# creat function to integrate mRNA levels within nuclei
filename = "2022_12_15 HCR Hand2 Tbx5a Fgf10a_1"
dataRoot = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/pecFin/HCR_Data/built_zarr_files/"
level = 0
mRNA_channels = [0, 1, 2]



####################
# load label dataset
label_path = dataRoot + filename + ".zarrlabels"

reader_lb = Reader(parse_url(label_path))

# nodes may include images, labels etc
nodes_lb = list(reader_lb())

# first node will be the image pixel data
label_node = nodes_lb[1]
label_data = label_node.data

####################
# load image dataset
image_path = dataRoot + filename + ".zarr"

reader_im = Reader(parse_url(image_path))

# nodes may include images, labels etc
nodes_im = list(reader_im())

# first node will be the image pixel data
image_node = nodes_im[0]
image_data = image_node.data

#####################
## extract key metadata

omero_attrs = image_node.root.zarr.root_attrs['omero']
channel_metadata = omero_attrs['channels']  # list of channels and relevant info
multiscale_attrs = image_node.root.zarr.root_attrs['multiscales']
scale_vec = multiscale_attrs[0]["datasets"][level]["coordinateTransformations"][0]["scale"]

channel_names = [channel_metadata[i]["label"] for i in mRNA_channels]

# Compute bp arrays from dask
label_array = label_data[level].compute()


# iterate through channels
regions = regionprops(label_array)



if __name__ == "__main__":
    # define some variables
    level = 1
    dataRoot = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/pecFin/HCR_Data/built_zarr_files/"
    labelRoot = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/pecFin/HCR_Data/built_zarr_files/"

    extract_nucleus_stats(dataRoot, level)