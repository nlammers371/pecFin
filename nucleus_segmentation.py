"""
Copyright 2022 (C)
    Friedrich Miescher Institute for Biomedical Research and
    University of Zurich

    Original authors:
    Tommaso Comparin <tommaso.comparin@exact-lab.it>
    Marco Franzon <marco.franzon@exact-lab.it>
    Joel Lüthi  <joel.luethi@fmi.ch>

    This file is part of Fractal and was originally developed by eXact lab
    S.r.l.  <exact-lab.it> under contract with Liberali Lab from the Friedrich
    Miescher Institute for Biomedical Research and Pelkmans Lab from the
    University of Zurich.

Image segmentation via Cellpose library
"""

import shutil
import logging
import glob2 as glob
import os
import time
from ome_zarr.io import parse_url
from skimage.transform import resize
from ome_zarr.reader import Reader
from typing import Any
from typing import Dict
from typing import Literal
from typing import Optional
import dask.array as da
import numpy as np
import zarr
from cellpose import models
from cellpose.core import use_gpu
import fractal_tasks_core
from fractal_tasks_core.lib_pyramid_creation import build_pyramid

logger = logging.getLogger(__name__)


__OME_NGFF_VERSION__ = fractal_tasks_core.__OME_NGFF_VERSION__


def segment_FOV(
    column: np.ndarray,
    model=None,
    do_3D: bool = True,
    anisotropy=None,
    diameter: float = 40.0,
    cellprob_threshold: float = 0.0,
    flow_threshold: float = 0.4,
    min_size=None,
    label_dtype=None,
    pretrain_flag=False
):
    """
    Internal function that runs Cellpose segmentation for a single ROI.

    :param column: Three-dimensional numpy array
    :param model: TBD
    :param do_3D: TBD
    :param anisotropy: TBD
    :param diameter: TBD
    :param cellprob_threshold: TBD
    :param flow_threshold: TBD
    :param min_size: TBD
    :param label_dtype: TBD
    """

    # Write some debugging info
    logger.info(
        f"[segment_FOV] START Cellpose |"
        f" column: {type(column)}, {column.shape} |"
        f" do_3D: {do_3D} |"
        f" model.diam_mean: {model.diam_mean} |"
        f" diameter: {diameter} |"
        f" flow threshold: {flow_threshold}"
    )

    # Actual labeling
    t0 = time.perf_counter()
    if not pretrain_flag:
        mask, flows, styles = model.eval(
            column,
            channels=[0, 0],
            do_3D=do_3D,
            net_avg=False,
            augment=False,
            diameter=diameter,
            anisotropy=anisotropy,
            cellprob_threshold=cellprob_threshold,
            flow_threshold=flow_threshold,
        )
    else:
        mask, flows, styles = model.eval(
            column,
            channels=[0, 0],
            do_3D=do_3D,
            min_size=min_size,
            net_avg=False,
            augment=False
        )
    if not do_3D:
        mask = np.expand_dims(mask, axis=0)
    t1 = time.perf_counter()

    # Write some debugging info
    logger.info(
        f"[segment_FOV] END   Cellpose |"
        f" Elapsed: {t1-t0:.4f} seconds |"
        f" mask shape: {mask.shape},"
        f" mask dtype: {mask.dtype} (before recast to {label_dtype}),"
        f" max(mask): {np.max(mask)} |"
        f" model.diam_mean: {model.diam_mean} |"
        f" diameter: {diameter} |"
        f" flow threshold: {flow_threshold}"
    )

    return mask.astype(label_dtype)


def cellpose_segmentation(
    *,
    # Fractal arguments
    zarr_directory: str,
    # Task-specific arguments
    level: int,
    seg_channel_label: Optional[str] = None,
    diameter_level0: float = 80.0,
    cellprob_threshold: float = 0.0,
    flow_threshold: float = 0.4,
    output_label_name: Optional[str] = None,
    model_type: Literal["nuclei", "cyto", "cyto2"] = "nuclei",
    pretrained_model: Optional[str] = None,
    overwrite: Optional[bool] = False
) -> Dict[str, Any]:
    """
    Run cellpose segmentation on the ROIs of a single OME-NGFF image

    Full documentation for all arguments is still TBD, especially because some
    of them are standard arguments for Fractal tasks that should be documented
    in a standard way. Here are some examples of valid arguments::

        input_paths = ["/some/path/*.zarr"]
        component = "some_plate.zarr/B/03/0"
        metadata = {"num_levels": 4, "coarsening_xy": 2}

    :param zarr_directory: path to directory containing zarr folders for images to segment
    :param level: Pyramid level of the image to be segmented.
    :param seg_channel_label: Identifier of a channel based on its label (e.g.
                          ``DAPI``). If not ``None``, then ``wavelength_id``
                          must be ``None``.
    :param diameter_level0: Initial diameter to be passed to
                            ``CellposeModel.eval`` method (after rescaling from
                            full-resolution to ``level``).
    :param output_label_name: output name for labels
    :param cellprob_threshold: Parameter of ``CellposeModel.eval`` method.
    :param flow_threshold: Parameter of ``CellposeModel.eval`` method.
    :param model_type: Parameter of ``CellposeModel`` class.
    :param pretrained_model: Parameter of ``CellposeModel`` class (takes
                             precedence over ``model_type``).
    """

    # Read useful parameters from metadata
    coarsening_xy = 2 #NL: need to store this in metadata
    min_size = (diameter_level0/coarsening_xy)**2
    # Preliminary check
    if seg_channel_label is None:
        raise ValueError(
            f"{seg_channel_label=} argument must be provided"
        )

    # get list of images
    image_list = sorted(glob.glob(zarr_directory + "*.zarr"))

    for im in range(len(image_list)):
        zarrurl = image_list[im]
        # read the image data
        store = parse_url(zarrurl, mode="r").store
        reader = Reader(parse_url(zarrurl))

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

        num_levels = len(dataset_info)

        # Find channel index
        ind_channel = None
        for ch in range(len(channel_metadata)):
            lbl = channel_metadata[ch]["label"]
            if lbl == seg_channel_label:
                ind_channel = ch

        if ind_channel == None:
            raise Exception(f"ERROR: Specified segmentation channel ({len(seg_channel_label)}) was not found in data")

        # Load ZYX data
        data_zyx = image_data[level][ind_channel, :, :, :]

        #data_zyx = da.from_zarr(f"{zarrurl}{level}")[ind_channel]
        logger.info(f"{data_zyx.shape=}")


        # Read pixel sizes from zattrs file
        full_res_pxl_sizes_zyx = dataset_info[0]["coordinateTransformations"][0]["scale"]
        actual_res_pxl_sizes_zyx = dataset_info[level]["coordinateTransformations"][0]["scale"]

        # calculate anisotropy
        anisotropy = actual_res_pxl_sizes_zyx[0]/actual_res_pxl_sizes_zyx[1]
        # resample z to make it isotropic
        us_factor = np.round(data_zyx.shape[0]*anisotropy)

        data_zyx = resize(data_zyx, (us_factor, data_zyx.shape[1], data_zyx.shape[2]), order=1, preserve_range=True)

        # Select 2D/3D behavior and set some parameters
        do_3D = data_zyx.shape[0] > 1

        # Prelminary checks on Cellpose model
        if pretrained_model is None:
            if model_type not in ["nuclei", "cyto2", "cyto"]:
                raise ValueError(f"ERROR model_type={model_type} is not allowed.")
        else:
            if not os.path.exists(pretrained_model):
                raise ValueError(f"{pretrained_model=} does not exist.")

        if output_label_name is None:
            try:
                omero_label = channel_metadata[ind_channel]["label"]
                output_label_name = f"label_{omero_label}"
            except (KeyError, IndexError):
                output_label_name = f"label_{ind_channel}"

        segment_flag = True
        if os.path.isdir(zarrurl+'labels') and overwrite:
            shutil.rmtree(zarrurl+'labels')
        elif os.path.isdir(zarrurl+'labels'):
            segment_flag = False

        if segment_flag:
            labels_group = zarr.group(f"{zarrurl}labels")
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

            # Open new zarr group for mask 0-th level
            logger.info(f"{zarrurl}labels/{output_label_name}/0")
            label_dtype = np.uint32 # NL: this may be way bigger than is actually required

            write_store = da.core.get_mapper(f"{zarrurl}labels/{output_label_name}/0")
            mask_zarr = zarr.create(
                shape=image_data[0][ind_channel, :, :, :].shape,
                chunks=image_data[0][ind_channel, :, :, :].chunksize,
                dtype=label_dtype,
                store=write_store,
                overwrite=False,
                dimension_separator="/",
            )

           # logger.info(
           #     f"mask will have shape {data_zyx.shape} "
           #     f"and chunks {data_zyx.chunks}"
           # )

            # Initialize cellpose
            gpu = use_gpu()
            if pretrained_model:
                model = models.CellposeModel(
                    gpu=gpu, pretrained_model=pretrained_model
                )
            else:
                model = models.CellposeModel(gpu=gpu, model_type=model_type)

            # Initialize other things
            logger.info(f"Start cellpose_segmentation task for {zarrurl}")
            logger.info(f"do_3D: {do_3D}")
            logger.info(f"use_gpu: {gpu}")
            logger.info(f"level: {level}")
            logger.info(f"model_type: {model_type}")
            logger.info(f"pretrained_model: {pretrained_model}")
            # logger.info(f"anisotropy: {anisotropy}")
            # logger.info(f"Total well shape/chunks:")
            # logger.info(f"{data_zyx.shape}")
            # logger.info(f"{data_zyx.chunks}")

            # Execute illumination correction
            image_mask = segment_FOV(
                data_zyx, #data_zyx.compute(),
                model=model,
                do_3D=do_3D,
                anisotropy=anisotropy,
                label_dtype=label_dtype,
                diameter=diameter_level0 / coarsening_xy**level,
                cellprob_threshold=cellprob_threshold,
                flow_threshold=flow_threshold,
                min_size=min_size,
                pretrain_flag=(pretrained_model != None)
            )

            # expand mask image to be at 0-th level resolution
            image_mask = image_mask.astype(float)
            shape0 = image_data[0][ind_channel, :, :, :].shape
            #image_mask_1 = resize(image_mask, (image_mask.shape[0], shape0[1], shape0[2]), order=0)
            image_mask_0 = resize(image_mask, shape0, order=0, anti_aliasing=False, preserve_range=True)
            #plt.imshow(image_mask_0[5,:,:], cmap='hot', interpolation='nearest')
            #plt.show()

            # Compute and store 0-th level to disk
            da.array(image_mask_0).to_zarr(
                url=mask_zarr,
                compute=True,
            )

            logger.info(
                f"End cellpose_segmentation task for {zarrurl}, "
                "now building pyramids."
            )

            # Starting from on-disk highest-resolution data, build and write to disk a
            # pyramid of coarser levels
            build_pyramid(
                zarrurl=f"{zarrurl}labels/{output_label_name}",
                overwrite=False,
                num_levels=num_levels,
                coarsening_xy=coarsening_xy,
                chunksize=image_data[0][ind_channel, :, :, :].chunksize,
                aggregation_function=np.max,
            )

            logger.info(f"End building pyramids, exit")
        else:
            print(f"WARNING: {zarrurl}labels already exists. Skipping. Set overwrite=True to overwrite")

    return {}

if __name__ == "__main__":
    zarr_directory = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/pecFin/HCR_Data/built_zarr_files_small/"

    # temporaily set parameters
    # zarrurl = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/pecFin/HCR_Data/built_zarr_files_testing/2022_12_15 HCR Hand2 Tbx5a Fgf10a_1.zarr"
    #output_path = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/pecFin/HCR_Data/segmented_zarr_files_testing/2022_12_15 HCR Hand2 Tbx5a Fgf10a_1.zarr"
    seg_channel_label = 'DAPI'
    level = 1
    pretrained_model = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/pecFin/HCR_Data/tiff_training_slices_testing/models/DAPI-Pro-2"
    overwrite = True
    model_type = "nuclei"
    output_label_name = "DAPI"
    test_flag = True

    diameter_level0 = 80
    cellprob_threshold = -0.5
    flow_threshold = 0.6  # NL: I don't think this is factored in for 3D
    min_size = diameter_level0 ** 3

    cellpose_segmentation(zarr_directory=zarr_directory, level=level, seg_channel_label=seg_channel_label,
                          output_label_name=output_label_name, pretrained_model=pretrained_model)


    """
    from pydantic import BaseModel
    from fractal_tasks_core._utils import run_fractal_task

    class TaskArguments(BaseModel):
        # Fractal arguments
        input_paths: Sequence[Path]
        output_path: Path
        component: str
        metadata: Dict[str, Any]
        # Task-specific arguments
        seg_channel_label: Optional[str] = None
        wavelength_id: Optional[str] = None
        level: int
        relabeling: bool = True
        anisotropy: Optional[float] = None
        diameter_level0: float = 80.0
        cellprob_threshold: float = 0.0
        flow_threshold: float = 0.4
        ROI_table_name: str = "FOV_ROI_table"
        bounding_box_ROI_table_name: Optional[str] = None
        output_label_name: Optional[str] = None
        model_type: Literal["nuclei", "cyto", "cyto2"] = "nuclei"
        pretrained_model: Optional[str] = None

    run_fractal_task(
        task_function=cellpose_segmentation,
        TaskArgsModel=TaskArguments,
        logger_name=logger.name,
    )

"""