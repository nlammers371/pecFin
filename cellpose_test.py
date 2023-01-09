from ome_zarr.io import parse_url
from ome_zarr.reader import Reader
import napari
import numpy as np
from fractal_tasks_core import cellpose_segmentation


input_path = ["/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/pecFin/HCR_Data/zarr_files/2022_12_15 HCR Hand2 Tbx5a Fgf10a_1.zarr"]
output_path = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/pecFin/HCR_Data/cellpose_test/2022_12_15 HCR Hand2 Tbx5a Fgf10a_1.zarr"


if __name__ == '__main__':
    napari.run()