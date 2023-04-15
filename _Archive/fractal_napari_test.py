"""
Copyright 2022 (C) Friedrich Miescher Institute for Biomedical Research and
University of Zurich

Original authors:
Marco Franzon <marco.franzon@exact-lab.it>
Tommaso Comparin <tommaso.comparin@exact-lab.it>

This file is part of Fractal and was originally developed by eXact lab S.r.l.
<exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
Institute for Biomedical Research and Pelkmans Lab from the University of
Zurich.
"""
import os
from pathlib import Path

from devtools import debug
from fractal_tasks_core import yokogawa_to_ome_zarr
from fractal_tasks_core import create_ome_zarr
from fractal_tasks_core import (
    napari_workflows_wrapper,
)

channel_parameters = {
    "A01_C01": {
        "label": "DAPI",
        "colormap": "00FFFF",
        "start": 0,
        "end": 700,
    },
    "A01_C02": {
        "label": "nanog",
        "colormap": "FF00FF",
        "start": 0,
        "end": 180,
    },
    "A02_C03": {
        "label": "Lamin B1",
        "colormap": "FFFF00",
        "start": 0,
        "end": 1500,
    },
}

num_levels = 6
coarsening_xy = 2


# Init
img_path = Path("../images/10.5281_zenodo.7059515/*.png")
if not os.path.isdir(img_path.parent):
    raise FileNotFoundError(
        f"{img_path.parent} is missing,"
        " try running ./fetch_test_data_from_zenodo.sh"
    )
zarr_path = Path("tmp_out/*.zarr")
metadata = {}

# Create zarr structure
metadata_update = create_ome_zarr.create_ome_zarr(
    input_paths=[img_path],
    output_path=zarr_path,
    allowed_channels=channel_parameters,
    num_levels=num_levels,
    coarsening_xy=coarsening_xy,
    metadata_table="mrf_mlf",
)
metadata.update(metadata_update)
debug(metadata)

# Yokogawa to zarr
for component in metadata["image"]:
    yokogawa_to_ome_zarr.yokogawa_to_ome_zarr(
        input_paths=[zarr_path],
        output_path=zarr_path,
        metadata=metadata,
        component=component,
    )
debug(metadata)

# napari-workflows
workflow_file = "wf_1.yaml"
input_specs = {
    "input": {"type": "image", "channel": "A01_C01"},
}
output_specs = {
    "Result of Expand labels (scikit-image, nsbatwm)": {
        "type": "label",
        "label_name": "label_DAPI",
    },
}
for component in metadata["image"]:
    napari_workflows_wrapper(
        input_paths=[zarr_path],
        output_path=zarr_path,
        metadata=metadata,
        component=component,
        input_specs=input_specs,
        output_specs=output_specs,
        workflow_file=workflow_file,
        ROI_table_name="FOV_ROI_table",
    )
debug(metadata)
