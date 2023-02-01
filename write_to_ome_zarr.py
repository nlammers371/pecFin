import os
import gc
import numpy as np
from aicsimageio import AICSImage
import dask
import shutil
from pathlib import Path
from pathlib import Path
from typing import Any
from ome_zarr.reader import Reader
from typing import Dict
import time
from typing import List
from typing import Sequence
import ome_zarr
import pandas as pd
import zarr
from ome_zarr.io import parse_url
from ome_zarr.writer import write_image
from anndata.experimental import write_elem
import skimage.io
import fractal_tasks_core

import glob2 as glob
import re

import logging

__OME_NGFF_VERSION__ = fractal_tasks_core.__OME_NGFF_VERSION__


def define_omero_channels(
        *,
        channels: Sequence[Dict[str, Any]],
        bit_depth: int,
        label_prefix: str = None,
) -> List[Dict[str, Any]]:
    """
    Update a channel list to use it in the OMERO/channels metadata

    Given a list of channel dictionaries, update each one of them by:
        1. Adding a set of OMERO-specific attributes;
        2. Discarding all other attributes.

    The ``new_channels`` output can be used in the
    ``attrs["omero"]["channels"]`` attribute of an image group.

    :param channels: A list of channel dictionaries (each one must include the
                     ``wavelength``  and ``gene`` keys).
    :param bit_depth: bit depth
    :returns: ``new_channels``, a new list of consistent channel dictionaries
              that can be written to OMERO metadata.

    """

    new_channels = []

    gene_colors: Dict[str, str] = {'Sox9a': 'blue', 'Hand2': 'green', 'Prdm1a': 'cyan',
                                            'Tbx5a': 'yellow', 'Myod1': 'bop orange', 'Robo3': 'I Orange',
                                            'Emilin3a': 'red', 'Fgf10a': 'magenta', 'Col11a2': 'I Purple',
                                            'DAPI': 'gray'}


    for channel in channels:
        wavelength = channel["wavelength"]

        # Always set a label
        label = channel["gene"]

        # assign color according to gene

        # Set colormap attribute. If not specificed, use the default ones (for
        # the first three channels) or gray
        colormap = gene_colors[label]

        # Set window attribute
        window = {
            "min": 0,
            "max": 2 ** bit_depth - 1,
        }

        #if "start" in channel.keys() and "end" in channel.keys():
        #    window["start"] = channel["start"]
        #    window["end"] = channel["end"]

        new_channel = {
            "label": label,
            "wavelength": wavelength,
            "active": True,
            "coefficient": 1,
            "color": colormap,
            "family": "linear",
            "inverted": False,
            "window": window,
        }
        new_channels.append(new_channel)

    # Check that channel labels are unique for this image
    labels = [c["label"] for c in new_channels]
    if len(set(labels)) < len(labels):
        raise ValueError(f"Non-unique labels in {new_channels=}")

    return new_channels

# define simple function to extract key metadata from experiment file name
def parse_experiment_name(f_name):

    fnames = f_name.split(" ")
    if fnames[1] == "HCR":
        date_string = fnames[0]
        # define HCR-specific gene-to-wavelength dictionary set up by Phil
        dict_gene_wavelength: Dict[str, str] = {'Sox9a': 'AF488-T3', 'Hand2': 'AF488-T3', 'Prdm1a': 'AF488-T3',
                                                'Tbx5a': 'AF546-T2', 'Myod1': 'AF546-T2', 'Robo3': 'AF546-T2',
                                                'Emilin3a': 'AF647-T1', 'Fgf10a': 'AF647-T1', 'Col11a2': 'AF647-T1'}

        wvl_list = ['DAPI-T4']
        gene_list = ['DAPI']
        for gene in fnames[2:5]:
            gene_list.append(gene)
            wvl_list.append(dict_gene_wavelength[gene])

    else:
        raise Exception(f"ERROR: {f_name} is not an HCR experiment")

    # enforce sort ordering by ascending wavelength
    #si = np.argsort(wvl_list)
    #wvl_list = [wvl_list[i] for i in si]
    #gene_list = [gene_list[i] for i in si]

    return gene_list, wvl_list


def write_to_ome_zarr(project_directory, write_directory, write_tiff=False, test_flag=False, num_levels=5, coarsening_factor=2, match_string='*', overwrite=False):
    """

    :param project_directory: [string] Path to folder containing experiment subfolders
    :param write_directory: [string] Path to destination folder
    :param write_tiff: [Logical] If true, code will also generate standard tiff stacks
    :param num_levels: [int] Number of coarse-graining levels to include in ome-zarr pyramid
    :param test_flag [logical] if true, data will be downsampled and saved to test directory
    :param coarsening_factor:  [int] Factor by which each successive level is downsampled
    :param match_string: [string] String to specify subset of folders within directory via string matching
    :param overwrite: [logical] Logical indicating whether or not to overwrite exisiting files. If False, code will skip to next file
    :return:
    """
    logger = logging.getLogger(__name__)

    # get list of experiment folders within the project directory
    experiment_list = glob.glob(project_directory + match_string)

    # initialize timer
    program_starts = time.time()

    for ex in range(len(experiment_list)):

        # extract subfolder names
        folder_name = experiment_list[ex].replace(project_directory, '', 1)
        #folder_name = "2022_12_15 HCR Hand2 Tbx5a Fgf10a"


        # get list of valid image files within directory
        image_read_dir = project_directory + folder_name + '/'
        image_list = sorted(glob.glob(image_read_dir + folder_name + "_?.czi"))

        ######################



        # initialize scaling method that reflects above options
        scaler_method = ome_zarr.scale.Scaler(downscale=coarsening_factor, max_layer=num_levels-1)

        # set paths to raw data
        for im in range(len(image_list)):
            #im = 0

            image_path = Path(image_list[im])
            image_name = image_list[im].replace(image_read_dir, '', 1)
            image_name = image_name.replace('.czi', '')

            # parse image name
            gene_list, wvl_list = parse_experiment_name(folder_name)

            #dict_plate_prefixes: Dict[str, Any] = {}

            info = (
                f"Listing all genes/channels from {image_path.as_posix()}\n"
                f"Genes:   {gene_list}\n"
                f"Channels: {wvl_list}\n"
            )

            # Check that we have 4 channels
            if len(gene_list) != 4:
                raise Exception(f"{info}ERROR: {len(gene_list)} channels/genes detected (expecting 4)")

            ################################################################
            # read in metadata
            imObject = AICSImage(image_path)

            # Extract pixel sizes and bit_depth
            res_raw = imObject.physical_pixel_sizes
            res_array = np.asarray(res_raw)
            res_array = np.insert(res_array, 0, 1)
            pixel_size_z = res_array[1]
            pixel_size_x = res_array[2]
            pixel_size_y = res_array[3]

            bit_depth_str = np.dtype(imObject.dask_data)
            if bit_depth_str == 'uint16':
                bit_depth = 16
            elif bit_depth_str == 'uint8':
                bit_depth = 8
            else:
                raise Exception(f"ERROR: {bit_depth_str} bit depth not recognized (expecting uint16 or uint8)")

            # get list of channel names
            channel_names = imObject.channel_names

            # generate list of channel dictionaries
            channel_dict_list = []
            channel_ro_list = []
            for channel in channel_names:
                channel_ind = wvl_list.index(channel)
                channel_ro_list.append(channel_ind)
                dict_entry: Dict[str, any] = {'wavelength': wvl_list[channel_ind],
                                              'gene': gene_list[channel_ind]}
                channel_dict_list.append(dict_entry)

            # Define image zarr
            if test_flag:
                outDir = write_directory + '/built_zarr_files_testing/'
            else:
                outDir = write_directory + '/built_zarr_files/'
            zarrurl = f"{outDir + image_name}.zarr"

            skip_flag = True

            if os.path.isdir(zarrurl) and not overwrite:

                # look confirm that all subdirectories are completed and non-empty
                subdir_list = sorted(glob.glob(zarrurl + "/" + "?"))
                if len(subdir_list) == num_levels:
                    # are all the directories non-empty?
                    for d in range(len(subdir_list)):
                        #print(subdir_list[d])
                        subsubdir_list = sorted(glob.glob(subdir_list[d] + "/?"))

                        skip_flag = len(subsubdir_list) > 0
                else:
                    skip_flag = False
            else:
                skip_flag = False

            # extract "top level" dimensions for pixel size calculations
            dask_data = dask.array.squeeze(imObject.dask_data)
            if test_flag:
                dask_data = dask_data[:, 75:105, 900:1412, 900:1412]
            if not skip_flag:
                if os.path.isdir(zarrurl):
                   shutil.rmtree(zarrurl)

                logger.info(f"Creating {zarrurl}")

                # write the image data and metadata to file
                store = parse_url(zarrurl, mode="w").store
                root = zarr.group(store=store)

                spatial_dims = dask_data.shape
                spatial_dims = spatial_dims[1:]

                trans_list = [
                             [
                                    {
                                        "type": "scale",
                                        "scale": [
                                            np.round(pixel_size_z, 4),
                                            np.round(pixel_size_y
                                                     * spatial_dims[1] / (
                                                         np.floor(spatial_dims[1] / coarsening_factor ** ind_level)), 4),
                                            np.round(pixel_size_x
                                                     * spatial_dims[2] / (
                                                         np.floor(spatial_dims[2] / coarsening_factor ** ind_level)), 4),
                                        ],
                                    }
                                ]
                            for ind_level in range(num_levels)
                        ],

                trans_list = trans_list[0]


                root.attrs["omero"] = {
                            "id": 1,  # NL: uncertain as to what this is meant to do
                            "name": "TBD",
                            "version": __OME_NGFF_VERSION__,
                            "channels": define_omero_channels(
                                channels=channel_dict_list, bit_depth=bit_depth
                            ),
                        }

                write_image(image=dask_data, group=root, scaler=scaler_method, axes="czyx", storage_options=dict(chunks=dask_data.chunksize))

                root.attrs["multiscales"] = [
                    {
                        "version": __OME_NGFF_VERSION__,
                        "axes": [
                            {"name": "c", "type": "channel"},
                            {
                                "name": "z",
                                "type": "space",
                                "unit": "micrometer",
                            },
                            {
                                "name": "y",
                                "type": "space",
                                "unit": "micrometer",
                            },
                            {
                                "name": "x",
                                "type": "space",
                                "unit": "micrometer",
                            },
                        ],
                        "datasets": [
                            {
                                "path": f"{ind_level}",
                                "coordinateTransformations": [
                                    {
                                        "type": "scale",
                                        "scale": [
                                            np.round(pixel_size_z, 4),
                                            np.round(pixel_size_y
                                                     * spatial_dims[1] / (
                                                         np.floor(spatial_dims[1] / coarsening_factor ** ind_level)), 4),
                                            np.round(pixel_size_x
                                                     * spatial_dims[2] / (
                                                         np.floor(spatial_dims[2] / coarsening_factor ** ind_level)), 4),
                                        ],
                                    }
                                ],
                            }
                            for ind_level in range(num_levels)
                        ],
                    }
                ]

            else:
                print(f"WARNING: {zarrurl} already exists. Skipping. Set overwrite=1 to overwrite")
                # raise Warning(f"WARNING: {zarrurl} already exists. Skipping. Set overwrite=1 to overwrite")

            # write tiff if desired
            if write_tiff:
                if test_flag:
                    tiffDir = write_directory + '/built_tiff_files_testing/'
                else:
                    tiffDir = write_directory + '/built_tiff_files/'

                filename = tiffDir + image_name
                if not os.path.isfile(filename) or overwrite:
                    np.random.seed(234)

                    if not os.path.isdir(tiffDir):
                        os.makedirs(tiffDir)

                    # load dopwn-sampled data
                    reader = Reader(parse_url(zarrurl))

                    # nodes may include images, labels etc
                    nodes = list(reader())

                    # first node will be the image pixel data
                    image_node = nodes[0]
                    np_data = np.asarray(image_node.data[1][3, :, :, :])

                    # save full image
                    skimage.io.imsave(filename + '.tiff', np_data)

                    logger.info(f"Creating {image_name} DAPI.tiff")

                    # save one XY, YZ, & XZ
                    zi = np.random.randint(0, np_data.shape[0])
                    xy_data = np_data[zi, :, :]
                    yi = np.random.randint(0, np_data.shape[1])
                    xz_data = np.squeeze(np_data[:, yi, :])
                    xi = np.random.randint(0, np_data.shape[2])
                    yz_data = np.squeeze(np_data[:, :, xi])

                    # save slices
                    skimage.io.imsave(filename + 'xy.tiff', xy_data)
                    skimage.io.imsave(filename + 'xz.tiff', xz_data)
                    skimage.io.imsave(filename + 'yz.tiff', yz_data)


            now = time.time()
            print("It has been {0} seconds since the loop started".format(now - program_starts))

            gc.collect()




if __name__ == '__main__':

    # def write_to_ome_zarr(project_directory, overwrite=False, match_string='*')
    overwrite = False
    match_string = '*'

    # set parameters
    num_levels = 5
    coarsening_factor = 2
    test_flag = True
    make_tiffs = True
    # set paths to raw data
    project_directory = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/pecFin/HCR_Data/raw/"
    # set write paths
    write_directory = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/pecFin/HCR_Data"

    # call main function
    write_to_ome_zarr(project_directory, write_directory, overwrite=True, test_flag=test_flag, write_tiff=make_tiffs)