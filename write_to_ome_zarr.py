import os
import numpy as np
from aicsimageio import AICSImage

from pathlib import Path
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Sequence

import pandas as pd
import zarr
from ome_zarr.io import parse_url
from ome_zarr.writer import write_image
from anndata.experimental import write_elem

import fractal_tasks_core
from fractal_tasks_core.lib_channels import check_well_channel_labels
from fractal_tasks_core.lib_channels import define_omero_channels
from fractal_tasks_core.lib_channels import validate_allowed_channel_input
from fractal_tasks_core.lib_metadata_parsing import parse_yokogawa_metadata
from fractal_tasks_core.lib_parse_filename_metadata import parse_filename
from fractal_tasks_core.lib_regions_of_interest import prepare_FOV_ROI_table
from fractal_tasks_core.lib_regions_of_interest import prepare_well_ROI_table
from fractal_tasks_core.lib_remove_FOV_overlaps import remove_FOV_overlaps

import glob2 as glob
import re

import logging

__OME_NGFF_VERSION__ = fractal_tasks_core.__OME_NGFF_VERSION__

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

logger = logging.getLogger(__name__)


# set paths to raw data
raw_path = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/pecFin/HCR_Data/raw/"
folder_name = "2022_12_15 HCR Hand2 Tbx5a Fgf10a"

# set write paths
write_folder = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/pecFin/HCR_Data"

# get list of valid image files within directory
image_read_dir = raw_path + folder_name + '/'
image_list = sorted(glob.glob(image_read_dir + folder_name + "_?.czi"))

######################
# set paths to raw data
im = 0
image_path = Path(image_list[im])
image_name = image_list[im].replace(image_read_dir,'',1)
image_name = image_name.replace('.czi','')

# parse image name
gene_list, wvl_list = parse_experiment_name(folder_name)

dict_plate_prefixes: Dict[str, Any] = {}

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
bit_depth = np.dtype(imObject.dask_data)

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
outDir = write_folder + '/built_zarr_files/'
zarrurl = f"{outDir + image_name}.zarr"
if os.path.isdir(outDir) != True:
   os.makedirs(outDir)

logger.info(f"Creating {zarrurl}")

# write the image data and metadata to file
store = parse_url(zarrurl, mode="w").store
root = zarr.group(store=store)

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
                                pixel_size_z,
                                pixel_size_y
                                * coarsening_xy ** ind_level,
                                pixel_size_x
                                * coarsening_xy ** ind_level,
                            ],
                        }
                    ],
                }
                for ind_level in range(num_levels)
            ],
        }
    ]

root.attrs["omero"] = {
            "id": 1,  # NL: uncertain as to what this is meant to do
            "name": "TBD",
            "version": __OME_NGFF_VERSION__,
            "channels": define_omero_channels(
                channels=actual_channels, bit_depth=bit_depth
            ),
        }
################################################################
for plate in plates:
    # Define plate zarr
    zarrurl = f"{plate}.zarr"
    image_path = dict_plate_paths[plate]
    logger.info(f"Creating {zarrurl}")
    group_plate = zarr.group(output_path.parent / zarrurl)
    zarrurls["plate"].append(zarrurl)

    # Obtain FOV-metadata dataframe

    if metadata_table == "mrf_mlf":
        mrf_path = f"{image_path}/MeasurementDetail.mrf"
        mlf_path = f"{image_path}/MeasurementData.mlf"
        site_metadata, total_files = parse_yokogawa_metadata(
            mrf_path, mlf_path
        )
        site_metadata = remove_FOV_overlaps(site_metadata)

    # If a metadata table was passed, load it and use it directly
    elif metadata_table.endswith(".csv"):
        site_metadata = pd.read_csv(metadata_table)
        site_metadata.set_index(["well_id", "FieldIndex"], inplace=True)

    # Extract pixel sizes and bit_depth
    pixel_size_z = site_metadata["pixel_size_z"][0]
    pixel_size_y = site_metadata["pixel_size_y"][0]
    pixel_size_x = site_metadata["pixel_size_x"][0]
    bit_depth = site_metadata["bit_depth"][0]

    if min(pixel_size_z, pixel_size_y, pixel_size_x) < 1e-9:
        raise Exception(pixel_size_z, pixel_size_y, pixel_size_x)

    # Identify all wells
    plate_prefix = dict_plate_prefixes[plate]

    plate_image_iter = glob(f"{image_path}/{plate_prefix}_{ext_glob_pattern}")

    wells = [
        parse_filename(os.path.basename(fn))["well"]
        for fn in plate_image_iter
    ]
    wells = sorted(list(set(wells)))

    # Verify that all wells have all channels
    for well in wells:
        well_image_iter = glob(
            f"{image_path}/{plate_prefix}_{well}{ext_glob_pattern}"
        )
        well_wavelength_ids = []
        for fpath in well_image_iter:
            try:
                filename_metadata = parse_filename(os.path.basename(fpath))
                well_wavelength_ids.append(
                    f"A{filename_metadata['A']}_C{filename_metadata['C']}"
                )
            except IndexError:
                logger.info(f"Skipping {fpath}")
        well_wavelength_ids = sorted(list(set(well_wavelength_ids)))
        if well_wavelength_ids != actual_wavelength_ids:
            raise Exception(
                f"ERROR: well {well} in plate {plate} (prefix: "
                f"{plate_prefix}) has missing channels.\n"
                f"Expected: {actual_channels}\n"
                f"Found: {well_wavelength_ids}.\n"
            )

    well_rows_columns = [
        ind for ind in sorted([(n[0], n[1:]) for n in wells])
    ]
    row_list = [
        well_row_column[0] for well_row_column in well_rows_columns
    ]
    col_list = [
        well_row_column[1] for well_row_column in well_rows_columns
    ]
    row_list = sorted(list(set(row_list)))
    col_list = sorted(list(set(col_list)))

    group_plate.attrs["plate"] = {
        "acquisitions": [{"id": 0, "name": plate}],
        "columns": [{"name": col} for col in col_list],
        "rows": [{"name": row} for row in row_list],
        "wells": [
            {
                "path": well_row_column[0] + "/" + well_row_column[1],
                "rowIndex": row_list.index(well_row_column[0]),
                "columnIndex": col_list.index(well_row_column[1]),
            }
            for well_row_column in well_rows_columns
        ],
    }

    for row, column in well_rows_columns:
        group_well = group_plate.create_group(f"{row}/{column}/")

        group_well.attrs["well"] = {
            "images": [{"path": "0"}],
            "version": __OME_NGFF_VERSION__,
        }

        group_image = group_well.create_group("0/")  # noqa: F841
        zarrurls["well"].append(f"{plate}.zarr/{row}/{column}/")
        zarrurls["image"].append(f"{plate}.zarr/{row}/{column}/0/")

        group_image.attrs["multiscales"] = [
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
                                    pixel_size_z,
                                    pixel_size_y
                                    * coarsening_xy ** ind_level,
                                    pixel_size_x
                                    * coarsening_xy ** ind_level,
                                ],
                            }
                        ],
                    }
                    for ind_level in range(num_levels)
                ],
            }
        ]

        group_image.attrs["omero"] = {
            "id": 1,  # FIXME does this depend on the plate number?
            "name": "TBD",
            "version": __OME_NGFF_VERSION__,
            "channels": define_omero_channels(
                channels=actual_channels, bit_depth=bit_depth
            ),
        }

        # Create tables zarr group for ROI tables
        group_tables = group_image.create_group("tables/")  # noqa: F841
        well_id = row + column

        # Prepare AnnData tables for FOV/well ROIs
        FOV_ROIs_table = prepare_FOV_ROI_table(site_metadata.loc[well_id])
        well_ROIs_table = prepare_well_ROI_table(
            site_metadata.loc[well_id]
        )

        # Write AnnData tables in the tables zarr group
        write_elem(group_tables, "FOV_ROI_table", FOV_ROIs_table)
        write_elem(group_tables, "well_ROI_table", well_ROIs_table)

    # Check that the different images in each well have unique channel labels.
    # Since we currently merge all fields of view in the same image, this check
    # is useless. It should remain there to catch an error in case we switch
    # back to one-image-per-field-of-view mode
    for well_path in zarrurls["well"]:
        check_well_channel_labels(
            well_zarr_path=str(output_path.parent / well_path)
        )

    metadata_update = dict(
        plate=zarrurls["plate"],
        well=zarrurls["well"],
        image=zarrurls["image"],
        num_levels=num_levels,
        coarsening_xy=coarsening_xy,
        original_paths=[str(p) for p in input_paths],
    )
    return metadata_update