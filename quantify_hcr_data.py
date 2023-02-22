import plotly.express as px
from ome_zarr.io import parse_url
import shutil
from ome_zarr.reader import Reader
from tqdm import tqdm
import pandas as pd
import numpy as np
import glob2 as glob
from skimage.measure import label, regionprops, regionprops_table
from scipy.ndimage import distance_transform_edt
from aicsimageio import AICSImage
import math
import dask
import os.path
import zarr
import dask.array as da
import fractal_tasks_core
from fractal_tasks_core.lib_pyramid_creation import build_pyramid
from skimage.transform import resize
from sklearn.neighbors import KDTree

__OME_NGFF_VERSION__ = fractal_tasks_core.__OME_NGFF_VERSION__

def calculate_cell_masks(label_array, labelPath, nucleus_df, image_data, scale_vec, level_in, multiscale_attrs, dataset_info,
                         num_levels=5, coarsening_xy=2, level_calc=1, output_label_name="CellCalc"):
    print("Calculating cell masks...")

    max_cell_extension = 2 # in um
    box_radius = 4
    chunk_size = (label_array.shape[0], round(431 / coarsening_xy ** level_calc), 
                  round(431 / coarsening_xy ** level_calc))

    # downsample to desired resolution for cell boundary calculations
    shape_lvl = image_data[level_calc][0, :, :, :].shape
    label_array_ds = resize(label_array, shape_lvl, order=0, anti_aliasing=False, preserve_range=True)
    
    scale_vec_ds = scale_vec
    lvl_diff = (level_calc-level_in)
    scale_vec_ds[1] = scale_vec_ds[1]*coarsening_xy**lvl_diff
    scale_vec_ds[2] = scale_vec_ds[2]*coarsening_xy**lvl_diff
    
    standard_radius = max_cell_extension / scale_vec_ds[1] #(0.75 * np.median(nucleus_df["Area"]) / np.pi) ** (1 / 3) / (coarsening_xy**(2*lvl_diff))*scale_vec_ds[0]/scale_vec_ds[1]
    max_radius = box_radius / scale_vec_ds[1]#(0.75 * np.max(nucleus_df["Area"]) / np.pi) ** (1 / 3) / (coarsening_xy ** (2 * lvl_diff)) * \
                      #scale_vec_ds[0] / scale_vec_ds[1]
    print("Maximum cell extension from nuclear mask: " + str(round(standard_radius)) + " pixels")
    nb_size_xy = round(2.5 * max_radius)
    nb_size_z = round(nb_size_xy * scale_vec_ds[1] / scale_vec_ds[0])
    d_shape = label_array_ds.shape
    label_array_cell = np.zeros(label_array_ds.shape)
    distance_array = np.ones(label_array_ds.shape) * np.inf

    for lb in range(nucleus_df.shape[0]):
        # extract small chunk from full array
        centroid = np.round(np.divide(nucleus_df[["Z", "Y", "X"]].iloc[lb], scale_vec_ds)).astype(int)
        zr = [np.max([centroid[0] - nb_size_z, 0]), np.min([centroid[0] + nb_size_z, d_shape[0]])]
        yr = [np.max([centroid[1] - nb_size_xy, 0]), np.min([centroid[1] + nb_size_xy, d_shape[1]])]
        xr = [np.max([centroid[2] - nb_size_xy, 0]), np.min([centroid[2] + nb_size_xy, d_shape[2]])]

        # calculate distance from nucleus boundary
        mask_chunk = label_array_ds[zr[0]:zr[1], yr[0]:yr[1], xr[0]:xr[1]] != (lb + 1)
        d_array = np.asarray(distance_transform_edt(mask_chunk))

        # compare to current distances
        dist_chunk = distance_array[zr[0]:zr[1], yr[0]:yr[1], xr[0]:xr[1]]
        lb_chunk = label_array_cell[zr[0]:zr[1], yr[0]:yr[1], xr[0]:xr[1]]

        # assign label to pixels that are closest to current cell (and within overall max radius)
        lb_pixels = np.where((d_array <= dist_chunk) & (d_array <= standard_radius))
        lb_chunk[lb_pixels] = lb + 1

        # set new minimums
        dist_chunk[lb_pixels] = d_array[lb_pixels]

        # map back to full array
        label_array_cell[zr[0]:zr[1], yr[0]:yr[1], xr[0]:xr[1]] = lb_chunk
        distance_array[zr[0]:zr[1], yr[0]:yr[1], xr[0]:xr[1]] = dist_chunk

    print("Done.")
    print("Saving cell masks to zarr pyramid...")
    if os.path.isdir(f"{labelPath}Cell/CellCalc"):
        shutil.rmtree(f"{labelPath}Cell/CellCalc")

    # add key metadata to file
    labels_group = zarr.group(f"{labelPath}Cell")
    labels_group.attrs["labels"] = [output_label_name]
    label_group = labels_group.create_group(output_label_name)
    label_group.attrs["image-label"] = {"version": __OME_NGFF_VERSION__}
    label_group.attrs["multiscales"] = [
        {
            "name": output_label_name,
            "version": __OME_NGFF_VERSION__,
            "axes": [
                ax for ax in multiscale_attrs[0]["axes"] if ax["type"] != "channel"
            ],
            "datasets": [
                dt for dt in dataset_info
            ]
        }
    ]

    write_store = da.core.get_mapper(f"{labelPath}Cell/{output_label_name}/0")

    # restore to base size (if necessary)
    shape0 = image_data[0][0, :, :, :].shape
    label_array_cell_0 = resize(label_array_cell, shape0, order=0, anti_aliasing=False, preserve_range=True)
    # Zarr label arrays must be integer type for napari to recognize them
    label_array_cell_0 = label_array_cell_0.astype(int)
    if level_in == 0:
        label_array_cell_out = label_array_cell_0
    else:
        label_array_cell_out = resize(label_array_cell, label_array.shape, order=0, anti_aliasing=False, preserve_range=True)

    # write labels to file
    mask_zarr = zarr.create(
        shape=label_array_cell_0.shape,
        chunks=chunk_size,
        dtype=label_array_cell_0.dtype,
        store=write_store,
        overwrite=False,
        dimension_separator="/",
    )

    da.array(label_array_cell_0).to_zarr(
        url=mask_zarr,
        compute=True,
    )

    # Starting from on-disk highest-resolution data, build and write to disk a
    # pyramid of coarser levels
    build_pyramid(
        zarrurl=f"{labelPath}Cell/{output_label_name}",
        overwrite=False,
        num_levels=num_levels,
        coarsening_xy=coarsening_xy,
        chunksize=chunk_size,
        aggregation_function=np.max,
    )
    print("Done.")
    return label_array_cell_out

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

def extract_nucleus_stats(dataRoot, rawRoot, level, mRNA_channels=None, resetClassGB=False, overwriteCellLabelsGB=False):

    # set default
    if mRNA_channels is None:
        mRNA_channels = [0, 1, 2]

    # get list of datasets with nucleus labels
    image_list = sorted(glob.glob(dataRoot + "*.zarrlabels"))

    for im in tqdm(range(len(image_list))):

        labelPath = image_list[im]
        labelName = image_list[im].replace(dataRoot, '', 1)
        cellLabelPath = labelPath + "Cell/"
        labelName = labelName.replace('.zarrlabels', '')
        rawImagePath = rawRoot + labelName[:-2] + "/" + labelName + ".czi" # have to draw from raw data for now...
        imagePath = dataRoot + labelName + ".zarr"
        savePath = dataRoot + labelName + '_nucleus_props.csv'

        resetClass = (not os.path.isfile(savePath)) | resetClassGB
        overwriteCellLabels = (not os.path.isdir(cellLabelPath)) | overwriteCellLabelsGB

        print('Extracting stats for ' + labelName)
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
        dataset_info = multiscale_attrs[0]['datasets']
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
        nucleus_df["Area"] = area_vec

        # calculate axis lengths
        axis_array = np.empty((len(regions), 3))
        vec_list = []
        for rgi, rg in enumerate(regions):
            moments = rg['moments_central']
            axes, axis_dirs = ellipsoid_axis_lengths(moments)
            axis_array[rgi, :] = np.multiply(axes, scale_vec)
            vec_list.append(axis_dirs)

        df2 = pd.DataFrame(axis_array, columns=["Axis_0", "Axis_1", "Axis_2"])
        df2["axis_dirs"] = vec_list

        nucleus_df = pd.concat([nucleus_df, df2], axis=1)

        if not resetClass:
            # check for class column in old dataset
            # load nucleus centroid data frame
            nucleus_df_orig = pd.read_csv(savePath, index_col=0)

            if 'pec_fin_flag' in nucleus_df_orig.columns:
                if nucleus_df.shape[0] == nucleus_df_orig.shape[0]:
                    nucleus_df['pec_fin_flag'] = nucleus_df_orig['pec_fin_flag']
                else:
                    raise Warning(
                        "Attempted to extract old pec fin assignments, but old and new datasets have different sizes. Skipping.")
            else:
                raise Warning(
                    "Attempted to extract old pec fin assignments, but 'pec_fin_flag' field does not exist. Skipping.")


        ################################
        # Calculate mRNA levels in each nucleus
        imObject = AICSImage(rawImagePath)
        image_array = imObject.data
        image_array = np.squeeze(image_array)
        #image_array = dask.array.rechunk(image_array, chunks='auto')
        #image_array = image_array.compute()

        # compute each channel of image array separately to avoid dask error
        im_array_list = []
        for ch in mRNA_channels:
            im_temp = image_array[ch, :, :, :]
            im_array_list.append(im_temp)

        # initialize empty columns
        print("Estimating cellular mRNA levels")
        omero_attrs = image_node.root.zarr.root_attrs['omero']
        channel_metadata = omero_attrs['channels']  # list of channels and relevant info
        channel_names = [channel_metadata[i]["label"] for i in mRNA_channels]
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
                mRNA_int = np.sum(im_ch[tuple(nc_coords.T)])
                nucleus_df.loc[rgi, channel_names[chi] + "_mean"] = mRNA_int / n_pix
                nucleus_df.loc[rgi, channel_names[chi]] = mRNA_int

        # normalize
        for ch in channel_names:
            int_max = np.max(nucleus_df[ch])
            mean_max = np.max(nucleus_df[ch + "_mean"])
            nucleus_df[ch] = nucleus_df.loc[:, ch] / int_max
            nucleus_df[ch + "_mean"] = nucleus_df.loc[:, ch + "_mean"] / mean_max


        #######################################################
        # expand nuclear masks to define cellular neighborhoods

        if overwriteCellLabels:
            label_array_cell = calculate_cell_masks(label_array, labelPath, nucleus_df, image_data, scale_vec, level,
                                                    multiscale_attrs, dataset_info)
        else:
            reader_lb_cell = Reader(parse_url(cellLabelPath))

            # nodes may include images, labels etc
            nodes_lb_cell = list(reader_lb_cell())

            # first node will be the image pixel data
            label_node_cell = nodes_lb_cell[1]
            label_array_cell = label_node_cell.data[level]

            label_array_cell = label_array_cell.compute()

        # now repeat above mRNA inference, this time using cell neighborhoods
        for ch in channel_names:
            nucleus_df[ch + "_cell"] = np.nan
            nucleus_df[ch + "_cell_mean"] = np.nan

        # calculate mRNA levels
        cell_regions = regionprops(label_array_cell)
        for rgi, rg in enumerate(cell_regions):
            # iterate through channels
            nc_coords = rg.coords.astype(int)
            n_pix = nc_coords.shape[0]
            for chi, im_ch in enumerate(im_array_list):
                # nc_ch_coords = np.concatenate((np.ones((n_pix,1))*ch, nc_coords), axis=1).astype(int)
                mRNA_int = np.sum(im_ch[tuple(nc_coords.T)])
                nucleus_df.loc[rgi, channel_names[chi] + "_cell_mean"] = mRNA_int / n_pix
                nucleus_df.loc[rgi, channel_names[chi] + "_cell"] = mRNA_int

        for ch in channel_names:
            int_max = np.max(nucleus_df.loc[:, ch + "_cell"])
            mean_max = np.max(nucleus_df.loc[:, ch + "_cell_mean"])
            nucleus_df.loc[:, ch + "_cell"] = nucleus_df.loc[:, ch + "_cell"] / int_max
            nucleus_df.loc[:, ch + "_cell_mean"] = nucleus_df.loc[:, ch + "_cell_mean"] / mean_max

        # remove empty rows (why do these appear?)
        nan_flag = np.isnan(nucleus_df.loc[:, "X"])
        nucleus_df = nucleus_df.loc[~nan_flag, :]

        # Average across nearest neighbors for both nucleus- and cell-based mRNA estimates
        nn_k = 6
        tree = KDTree(nucleus_df[["Z", "Y", "X"]], leaf_size=2)
        nearest_dist, nearest_ind = tree.query(nucleus_df[["Z", "Y", "X"]], k=nn_k + 1)

        for ch in channel_names:
            nucleus_df[ch + "_cell_nn"] = np.nan
            nucleus_df[ch + "_cell_mean_nn"] = np.nan
            nucleus_df[ch + "_nn"] = np.nan
            nucleus_df[ch + "_mean_nn"] = np.nan

        for row in range(nucleus_df.shape[0]):
            nearest_indices = nearest_ind[row, :]
            nn_df_temp = nucleus_df.iloc[nearest_indices]

            for ch in channel_names:
                int_mean = np.mean(nn_df_temp.loc[:, ch])
                mean_mean = np.mean(nn_df_temp.loc[:, ch + "_mean"])
                int_mean_cell = np.mean(nn_df_temp.loc[:, ch + "_cell"])
                mean_mean_cell = np.mean(nn_df_temp.loc[:, ch + "_cell_mean"])

                nucleus_df.loc[row, ch + "_cell_nn"] = mean_mean_cell
                nucleus_df.loc[row, ch + "_cell_mean_nn"] = int_mean_cell
                nucleus_df.loc[row, ch + "_nn"] = int_mean
                nucleus_df.loc[row, ch + "_mean_nn"] = mean_mean

        nucleus_df.to_csv(savePath)

if __name__ == "__main__":
    # define some variables
    level = 0
    dataRoot = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/pecFin/HCR_Data/built_zarr_files/"
    rawRoot = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/pecFin/HCR_Data/raw/"

    extract_nucleus_stats(dataRoot, rawRoot, level=0)
