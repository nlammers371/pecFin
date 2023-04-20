import pandas as pd
import numpy as np
import glob2 as glob
from sklearn.manifold import Isomap
import os
import matplotlib.pyplot as plt
from matplotlib import animation
import time
import scipy
from sklearn.linear_model import LinearRegression


def sphereFit_fixed_r(spX, spY, spZ, r0):
    #   Assemble the A matrix
    spX = np.array(spX)
    spY = np.array(spY)
    spZ = np.array(spZ)

    xyz_array = np.zeros((len(spX), 3))
    xyz_array[:, 0] = spX
    xyz_array[:, 1] = spY
    xyz_array[:, 2] = spZ
    c0 = np.mean(xyz_array, axis=0)
    c0[2] = -r0

    def ob_fun(c0, xyz=xyz_array, r=r0):
        res = np.sqrt((xyz[:, 0] - c0[0]) ** 2 + (xyz[:, 1] - c0[1]) ** 2 + (xyz[:, 2] - c0[2]) ** 2) - r
        return res

    C = scipy.optimize.least_squares(ob_fun, c0, bounds=([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, 0]))

    return r0, C.x[0], C.x[1], C.x[2]
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

def flatten_fin(df, k_nn=5, dv_thresh=30):
    """

    :param df: [pandas DataFrame] main data frame containing nucleus information
    :param k_nn: [int] number of nearest neighbors to consider for isomap calculations
    :param dv_thresh: nuclei further than this from the reference surface will be excluded
    :return: df: [DataFrame] with "flattened" position fields added
    """

    # get indices of fin nuclei
    fin_indices = np.where(df["pec_fin_flag"] == 2)[0]
    df_fin = df.iloc[fin_indices]

    # remove distant nuclei
    dv_filter = np.where(np.abs(df_fin["DV_pos"]) <= dv_thresh)
    df_fin = df_fin.iloc[dv_filter]
    fin_indices = fin_indices[dv_filter]
    #####
    # first, correct the PD position variable
    # fit the function
    base_nuclei = np.where(df["pec_fin_flag"] == 1)[0]
    xyz_base = df[["X", "Y", "Z"]].iloc[base_nuclei]
    xyz_base = xyz_base.to_numpy()
    # print(xyz_base)
    r, x0, y0, z0 = sphereFit_fixed_r(xyz_base[:, 0], xyz_base[:, 1], xyz_base[:, 2], 250)
    c0 = np.asarray([x0, y0, z0])


    pd_vec = df_fin["PD_pos"].to_numpy()

    surf_array = df_fin[["X_surf", "Y_surf", "Z_surf"]].to_numpy()
    dist_array = np.sqrt(np.sum((c0 - surf_array) ** 2, axis=1))
    outside_indices = np.where(dist_array >= r)[0]
    inside_indices = np.where(dist_array < r)[0]
    pd_vec[outside_indices] = np.abs(pd_vec[outside_indices])
    pd_vec[inside_indices] = -np.abs(pd_vec[inside_indices])

    # assign to dataframe
    df_fin["PD_pos"] = pd_vec
    df["PD_pos"].iloc[fin_indices] = pd_vec

    #####################
    # Apply Isomap transformation
    # transform
    embedding = Isomap(n_components=2, n_neighbors=k_nn)
    surf_transformed = embedding.fit_transform(surf_array)

    # correct it to align with biologically meaningful axes
    # use linear regression to back out correspondence between PD axis and the isomap dimensions
    reg = LinearRegression().fit(surf_transformed, pd_vec)

    new_axis1 = reg.coef_
    new_axis1 = new_axis1 / np.sqrt(np.sum((new_axis1 ** 2)))
    new_axis2 = [-new_axis1[1], new_axis1[0]]

    surf_tr_new = np.empty((surf_transformed.shape))
    surf_tr_new[:, 0] = new_axis1[0] * surf_transformed[:, 0] + new_axis1[1] * surf_transformed[:, 1]
    surf_tr_new[:, 1] = new_axis2[0] * surf_transformed[:, 0] + new_axis2[1] * surf_transformed[:, 1]

    # center the AP axis 
    surf_tr_new[:, 1] - np.mean(surf_tr_new[:, 1], axis=0)
    
    # assign 
    df["X_AP_st"] = np.nan
    df["X_AP_st"].iloc[fin_indices] = surf_tr_new[:, 1]
    df["Y_DV_st"] = np.nan
    df["Y_DV_st"].iloc[fin_indices] = df_fin["DV_pos"].copy()
    df["Z_PD_st"] = np.nan
    df["Z_PD_st"].iloc[fin_indices] = surf_tr_new[:, 0]

    return df


if __name__ == '__main__':
    dataRoot = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/pecFin/HCR_Data/nucleus_props/"

    # get list of filepaths
    fileList = sorted(glob.glob(dataRoot + '*_nucleus_props.csv'))
    pdNameList = []
    for fn in range(len(fileList)):
        filePath = fileList[fn]
        labelName = fileList[fn].replace(dataRoot, '', 1)
        labelName = labelName.replace('_nucleus_props.csv', '')
        pdNameList.append(labelName)

        if os.path.isfile(filePath):
            df = pd.read_csv(filePath, index_col=0)
        else:
            raise Exception(
                f"Selected dataset( {filePath} ) dataset has no nucleus data. Have you run extract_nucleus_stats?")

        df = flatten_fin(df)

        df.to_csv(filePath)


