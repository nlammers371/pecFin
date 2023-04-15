import dash
import plotly.express as px
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import json
import glob2 as glob
from skimage.measure import regionprops
import itertools
import os
import matplotlib.pyplot as plt
from matplotlib import animation
import time
from scipy.interpolate import LinearNDInterpolator
import alphashape


def load_nucleus_dataset(dataRoot, filename):
    # global fin_points_prev, not_fin_points_prev, class_predictions_curr, df, curationPath, propPath

    propPath = dataRoot + filename + '_nucleus_props.csv'

    if os.path.isfile(propPath):
        df = pd.read_csv(propPath, index_col=0)
    else:
        raise Exception(
            f"Selected dataset( {filename} ) dataset has no nucleus data. Have you run extract_nucleus_stats?")

#     fin_nuclei = np.where(df["pec_fin_flag"] == 1)
#     df = df.iloc[fin_nuclei]

    # normalize gene expression levels
    colnames = df.columns
    list_raw = [item for item in colnames if "_cell_mean_nn" in item]
    gene_names = [item.replace("_cell_mean_nn", "") for item in list_raw]

    for g in gene_names:
        ind_list = [i for i in range(len(colnames)) if g in colnames[i]]
        for ind in ind_list:
            colname = colnames[ind]
            c_max = np.max(df[colname])
            df[colname] = df.loc[:, colname] / c_max

    return df, gene_names


if __name__ == '__main__':
    dataRoot = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/pecFin/HCR_Data/nucleus_props/"

    figureRoot = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/pecFin/HCR_figures/"
    if os.path.isdir(figureRoot) == False:
        os.makedirs(figureRoot)

    # get list of filepaths
    fileList = sorted(glob.glob(dataRoot + '*_nucleus_props.csv'))
    pdNameList = []
    for fn in range(len(fileList)):
        labelName = fileList[fn].replace(dataRoot, '', 1)
        labelName = labelName.replace('_nucleus_props.csv', '')
        pdNameList.append(labelName)


    # make rotating figure
    df_ind = 2

    fileName = fileList[df_ind]
    imageName = pdNameList[df_ind]

    df, gene_name_dict = load_nucleus_dataset(dataRoot, imageName)
    df_fin = df.iloc[np.where(df["pec_fin_flag"] == 2)]

    fig = plt.figure()
    # ax.scatter(df["X"], df["Y"], df["Z"], alpha=0.5)
    base_range = range(90, 450, 10)
    angle_list = list(base_range) + list(reversed(base_range[:-1]))

    def animate(i, angle_list=angle_list):
        ax = fig.add_subplot(projection='3d')
        ax.scatter(df["X"], df["Y"], df["Z"], alpha=0.5)
        ax.view_init(elev=10., azim=angle_list[i])
        ax.axis("off")
        return fig,

    start = time.time()
    anim = animation.FuncAnimation(fig, animate,
                                   blit=True, interval=20, frames=len(angle_list))

    plt.close()
    # HTML(anim.to_html5_video())

    anim.save(figureRoot + imageName + "full_set_rotation.mp4", dpi=300, fps=10)
    end = time.time()

    print(end-start)