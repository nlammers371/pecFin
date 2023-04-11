import dash
import plotly.express as px
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import timeit
from pyntcloud import PyntCloud
import open3d as o3d
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader
import alphashape
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import json
import glob2 as glob
from sklearn.decomposition import PCA
import open3d as o3d
from pyntcloud import PyntCloud
from sklearn.cluster import KMeans
import itertools
from scipy.spatial import distance_matrix
from sklearn.neighbors import KDTree
import pickle
import os.path
import os
import networkx as nx
import math
import scipy
from scipy.optimize import fsolve
from sklearn.ensemble import RandomForestClassifier
from itertools import product
from scipy.integrate import quad


def lagrange_equations(P, FDECAB, P0):
    x, y, z, L = P  # varibles to solve for
    x0, y0, z0 = P0  # coordinates of point (known)
    F, D, E, C, A, B = FDECAB  # curve parameters (known)

    # dfdx
    dfdx = D * L + 2 * (1 + A * L) * x - 2 * x0 + C * L * y
    # dfdy
    dfdy = E * L + C * L * x + 2 * (y + B * L * y - y0)
    # dfdz
    dfdz = 2 * z - 2 * z0 - L
    # dfdL
    dfdl = F + D * x + A * x ** 2 + y * (E + C * x + B * y) - z

    return (dfdx, dfdy, dfdz, dfdl)

def numerical_integral2(t, FDECAB, P0, V):
    F, D, E, C, A, B = FDECAB
    v0, v1 = V
    x0, y0, _ = P0
    integrand = np.sqrt(v0**2 + v1**2 + (D*v0 + 2*A*v0*(t*v0 + x0) + v1*(E + 2*C*t*v0 + 2*B*t*v1 + C*x0) + (C*v0 + 2*B*v1)*y0)**2)

    return integrand
def numerical_integral(s, FDECAB, m, b):
    F, D, E, C, A, B = FDECAB
    integrand = np.sqrt(1 + m ** 2 + (D + E * m + b * (C + 2 * B * m) + 2 * (A + m * (C + B * m)) * s) ** 2)
    return integrand

# helper function for calculating minimum distance between point and surface
def definite_integral(FDECAB, m, b, s):
    F, D, E, C, A, B = FDECAB
    integralVal = ((D + E*m + b*(C + 2*B*m) + 2*(A + m*(C + B*m))*s)*
                    np.sqrt(1 + m**2 + (D + E*m + b*(C + 2*B*m) + 2*(A + m*(C + B*m))*s)**2) +
                    (1 + m**2)*np.arctanh((D + E*m + b*(C + 2*B*m) + 2*(A + m*(C + B*m))*s)/
                    np.sqrt(1 + m**2 + (D + E*m + b*(C + 2*B*m) + 2*(A + m*(C + B*m))*s)**2)))/(4.*(A + m*(C + B*m)))

    return integralVal

def calculate_path_distance(FDECAB, p0, p1):
    # F, D, E, C, A, B = FDECAB
    dx = p0[0] - p1[0]
    dy = p0[1] - p1[1]
    # m = dy/dx
    # b = p0[1] - m*p0[0]
    # T = np.sqrt(dx**2 + dy**2)

    I = quad(numerical_integral2, 0, 1, args=(FDECAB, p0, [dx, dy]))

    # v0, v1 = quad(numerical_integral, p0[0], p0[1], args=(FDECAB, m, b))
    # v0 = definite_integral(FDECAB, m, b, p0[0])
    # v1 = definite_integral(FDECAB, m, b, p1[0])
    return I[1] - I[0]


def calculate_tangent_plane2(DECA, base_point, sphere_point):
    D, E, C, A = DECA

    xb, yb, zb = base_point
    xs, ys, zs = sphere_point

    # calculate two components of normal
    a = 2 * A*xb + C*yb + D
    c = -1

    # solve for third
    b = (a*(-xb+xs) + c*(-zb+zs))/(yb-ys) # solution for b that forces plane to pass through both points
    B = (b-C*xb-E)/(2*yb) # B for quadratic surface such that its tangent plane at bp passes through sp

    plane_vec_norm = np.asarray([a, b, c])
    plane_vec_norm = plane_vec_norm / np.sqrt(np.sum(plane_vec_norm ** 2))
    # calculate D
    D = -np.dot(plane_vec_norm, base_point)

    return plane_vec_norm, D, B

def calculate_tangent_plane(C_full, point):
    F, D, E, C, A, B = C_full
    dfdx = 2 * A * point[0] + C * point[1] + D
    dfdy = 2 * B * point[1] + C * point[0] + E
    dfdz = -1
    plane_vec_norm = np.asarray([dfdx, dfdy, dfdz])
    plane_vec_norm = plane_vec_norm / np.sqrt(np.sum(plane_vec_norm ** 2))
    # calculate D
    D = -np.dot(plane_vec_norm, point)

    return (plane_vec_norm, D)

# Helper function to calculate F in surface equation given a point that the surface must pass through
def solve_for_f(C_fit, cp):
    # C0 = z0 - C1*x0 - C2*y0 - C3*x0*yz - C4*x0**2 - C5*Y0**2
    x0 = cp[0]
    y0 = cp[1]
    z0 = cp[2]
    C0 = z0 - C_fit[0] * x0 - C_fit[1] * y0 - C_fit[2] * x0 * y0 - C_fit[3] * x0 ** 2 - C_fit[4] * y0 ** 2
    C_out = [C0, C_fit[0], C_fit[1], C_fit[2], C_fit[3], C_fit[4]]
    return C_out

def predict_quadratic_surface2(xy, DECA, base_point, sphere_point):

    # first, solve for B given constraint that tangent plane must pass thrhoug sphere center
    plane_vec_norm, D, B = calculate_tangent_plane2(DECA, base_point, sphere_point)
    DECAB = [DECA[0], DECA[1], DECA[2], DECA[3], B]

    # then solve for F
    C_full = solve_for_f(DECAB, base_point)

    # predict surface
    A = np.c_[np.ones(xy.shape[0]), xy, np.prod(xy, axis=1), xy**2]
    surf_pd = np.dot(A, C_full).ravel()
    xyz_out = np.concatenate((xy, np.reshape(surf_pd, (xy.shape[0], 1))), axis=1)

    return xyz_out, C_full, plane_vec_norm, D

def predict_quadratic_surface(xy, DECAB, point):
    C_full = solve_for_f(DECAB, point)
    A = np.c_[np.ones(xy.shape[0]), xy, np.prod(xy, axis=1), xy**2]
    surf_pd = np.dot(A, C_full).ravel()
    xyz_out = np.concatenate((xy, np.reshape(surf_pd, (xy.shape[0], 1))), axis=1)

    return xyz_out, C_full
def fit_quadratic_surface(df, c0, pec_fin_nuclei, r):
    # fit corresponding plane
    # get nn distances
    k_nn = 7
    xyz_array = df[["X", "Y", "Z"]].to_numpy()
    tree = KDTree(xyz_array)
    nearest_dist, nearest_ind = tree.query(xyz_array, k=k_nn + 1)

    # find average distance to kth closest neighbor
    mean_nn_dist_vec = np.mean(nearest_dist, axis=0)
    nn_thresh = mean_nn_dist_vec[k_nn]
    nn_thresh_small = mean_nn_dist_vec[1]

    # find pec fin nuclei that are near the sphere surface
    xyz_fin = df[["X", "Y", "Z"]].iloc[pec_fin_nuclei].to_numpy()
    dist_array = np.sqrt(np.sum((c0 - xyz_fin) ** 2, axis=1)) - r
    surf_indices = np.where(dist_array <= nn_thresh)[0]
    xyz_surf = xyz_fin[surf_indices]

    # calculate centroid
    surf_cm = np.mean(xyz_surf, axis=0)

    # calculate PCA
    pca = PCA(n_components=3)

    # xyz_grid = np.concatenate((xyz_surf[:, 0], xyz_surf[:, 1], xyz_surf[:, 2), axis=1)
    pca.fit(xyz_surf)
    vec1 = pca.components_[0]
    vec2 = surf_cm - c0
    vec2 = vec2 / np.sqrt(np.sum(vec2 ** 2))
    plane_normal = np.cross(vec1, vec2)
    plane_normal_u = plane_normal / np.sqrt(np.sum(plane_normal ** 2))
    D_plane = -np.dot(plane_normal, surf_cm)

    # generate additional points
    c_vec = surf_cm - c0
    c_vec_u = c_vec / np.sqrt(np.sum(c_vec ** 2))
    lower_point = surf_cm - 2*c_vec_u*nn_thresh_small
    n_reps = 6
    add_array = np.empty((n_reps*n_reps+1, 3))
    add_array[0, :] = lower_point
    for n in range(n_reps):
        ind = 2*n
        add_array[ind+1, :] = lower_point + nn_thresh*vec1*(n+1)
        add_array[ind+2:, :] = lower_point - nn_thresh*vec1*(n+1)

    #######################
    # fit surface to fin points
    #######################
    xyz_fin_raw = df[["X", "Y", "Z"]].iloc[pec_fin_nuclei]
    xyz_surf = xyz_fin_raw.iloc[surf_indices].to_numpy()

    # exclude base points that are too far from our plane prior. If included, these will tend to cause the
    # surface fit to perform poorly
    d_surf = np.abs(np.sum(np.multiply(plane_normal_u, xyz_surf), axis=1) + D_plane)
    ind_rm = surf_indices[np.where(d_surf > nn_thresh)[0]]
    keep_indices = [k for k in range(xyz_fin_raw.shape[0]) if k not in ind_rm]
    xyz_fin = xyz_fin_raw.iloc[keep_indices].copy()

    # convert to point cloud and downsample points
    tree = KDTree(xyz_fin.to_numpy())
    nearest_dist, nearest_ind = tree.query(xyz_fin.to_numpy(), k=2)

    # find average distance to kth closest neighbor
    mean_nn_dist_vec = np.mean(nearest_dist, axis=0)
    nn_thresh1 = mean_nn_dist_vec[1]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_fin.to_numpy())
    pcd_down = pcd.voxel_down_sample(voxel_size=nn_thresh1 * 1.5)
    xyz_fin_down = np.asarray(pcd_down.points)

    # Transform to PCA space
    pca = PCA(n_components=3)
    pca.fit(xyz_fin_down)
    pca_fin_full = pca.transform(xyz_fin_raw.to_numpy())
    pca_fin_raw = pca.transform(xyz_fin_down)
    pca_add_array = pca.transform(add_array)
    pca_fin_surf = pca.transform(xyz_surf)

    # use kmeans to find key points
    n_clusters = 100
    kmeans = KMeans(n_clusters=n_clusters).fit(pca_fin_raw[:, 0:2])
    k_labels = kmeans.labels_
    pca_fin = np.empty((n_clusters, 3))
    for n in range(n_clusters):
        k_indices = np.where(k_labels == n)[0]
        pca_fin[n, :] = np.mean(pca_fin_raw[k_indices, :], axis=0)

    pca_fin = np.concatenate((pca_fin, pca_add_array), axis=0)
    # pca_fin = kmeans.cluster_centers_

    pca_surf_cm = pca.transform(np.reshape(surf_cm, (1, 3)))
    pca_surf_cm = pca_surf_cm.ravel()

    pca_c0 = pca.transform(np.reshape(c0, (1, 3)))
    pca_c0 = pca_c0.ravel()
    
    # print(pca_surf_cm)
    # generate grid to be used for surface generation
    grid_res = 25
    P0, P1 = np.meshgrid(np.linspace(np.min(pca_fin_raw[:, 0]), np.max(pca_fin_raw[:, 0]), grid_res),
                         np.linspace(np.min(pca_fin_raw[:, 1]), np.max(pca_fin_raw[:, 1]), grid_res))

    PP0 = P0.flatten()
    PP1 = P1.flatten()

    # def l2_fit_loss_point(C_prop, xy=pca_fin[:, 0:2], pp2=pca_fin[:, 2].ravel(), cp=pca_surf_cm):
    #     xyz_pd = predict_quadratic_surface(xy, C_prop, cp)
    #     residuals = xyz_pd[:, 2] - pp2
    #     return residuals

    xy2 = np.concatenate((PP0[:, np.newaxis], PP1[:, np.newaxis]), axis=1)
    def l2_fit_loss_point_euc(C_prop, xyz_fin=pca_fin, xy=xy2, cp=pca_surf_cm):
        xyz_pd, C0_full = predict_quadratic_surface(xy, C_prop, cp)
        dist_array = distance_matrix(xyz_fin, xyz_pd)
        residuals = np.min(dist_array, axis=1)
        return residuals

    def l2_fit_loss_plane_euc(C_prop, xyz_fin=pca_fin, xy=xy2, cp=pca_surf_cm, sp=pca_c0):
        xyz_pd, C0_full, _, _ = predict_quadratic_surface2(xy, C_prop, cp, sp)
        dist_array = distance_matrix(xyz_fin, xyz_pd)
        residuals = np.min(dist_array, axis=1)
        return residuals

    c0 = [-3, 0.01, 0.015, 0.001, 0.005, 0.01]
    c_fit = scipy.optimize.least_squares(l2_fit_loss_point_euc, c0[1:])
    C_fit = c_fit.x

    C_out = solve_for_f(C_fit, pca_surf_cm)
    # xyz_pd, C, normal_vec, D = predict_quadratic_surface2(xy2, C_fit, pca_surf_cm, pca_c0)
    PP2_curve = np.dot(np.c_[np.ones(PP0.shape), PP0, PP1, PP0 * PP1, PP0 ** 2, PP1 ** 2], C_out).reshape(P0.shape)

    xyz_fit_curve = pca.inverse_transform(np.concatenate((np.reshape(P0, (P0.size, 1)),
                                                          np.reshape(P1, (P1.size, 1)),
                                                          np.reshape(PP2_curve, (PP2_curve.size, 1))),
                                                         axis=1)
                                          )

    normal_vec, D = calculate_tangent_plane(C_out, pca_surf_cm)
    plane_pd = -(normal_vec[0] * P0 + normal_vec[1] * P1 + D) / normal_vec[2]


    xyz_plane = pca.inverse_transform(np.concatenate((np.reshape(P0, (P0.size, 1)),
                                                      np.reshape(P1, (P1.size, 1)),
                                                      np.reshape(plane_pd, (plane_pd.size, 1))),
                                                      axis=1)
                                      )

    ##########
    # sample points that fall approximately at the intersection of sphere and surface
    grid_res = 100
    P0B, P1B = np.meshgrid(np.linspace(np.min(pca_fin_surf[:, 0]), np.max(pca_fin_surf[:, 0]), grid_res),
                           np.linspace(np.min(pca_fin_surf[:, 1]), np.max(pca_fin_surf[:, 1]), grid_res))
    PP0B = P0B.flatten()
    PP1B = P1B.flatten()
    P2B = np.dot(np.c_[np.ones(PP0B.shape), PP0B, PP1B, PP0B * PP1B, PP0B ** 2, PP1B ** 2], C_out).reshape(P0B.shape)
    PP2B = P2B.flatten()

    pca_pd_array = np.concatenate((PP0B[:, np.newaxis], PP1B[:, np.newaxis], PP2B[:, np.newaxis]), axis=1)
    xyz_pd_array = pca.inverse_transform(pca_pd_array)

    ############
    # Calculate AP position of each base point. These will be used to assign AP positions to fin nuclei
    ##############
    # Now, calculate PD position for each fin nucleus
    # first, find points at intersection of surface and sphere
    sp_dist_array = np.abs(np.sqrt(np.sum((pca_pd_array - pca_c0) ** 2, axis=1)) - r)
    # print(xyz_pd_array)
    close_indices = np.where(sp_dist_array <= 1)[0]
    pca_pd_base = pca_pd_array[close_indices]

    # find point that is farthest away from plane defined by known points
    pca_b = PCA(n_components=3)
    pca_b.fit(pca_pd_base)
    pca_pca_b = pca_b.transform(pca_pd_base)
    ap_ind1 = np.argmin(pca_pca_b[:, 0])

    # identify point to use as reference
    # ap_ind1 = np.argmin(D_scores)
    ap_ref_point = pca_pd_base[ap_ind1, :]
    # now, find pd distance for each projected surface point
    ap_pos_base = np.empty((pca_pd_base.shape[0],))

    for a in range(ap_pos_base.shape[0]):
        ap_pos_base[a] = calculate_path_distance(FDECAB=C_out, p0=ap_ref_point, p1=pca_pd_base[a, :])

    ############
    # Find closest point to each nucleus that is on the inferred surface
    pca_point_surf = np.empty((pca_fin_full.shape))
    for p in range(pca_fin_full.shape[0]):
        Pinit = pca_fin_full[p, :]
        # def call_lagrange(P, P0=Pinit, FDECAB=C_out):
        #     x, y, z, L = lagrange_equations(P, P0=P0, FDECAB=FDECAB)
        #     return (x, y, z, L)

        def z_fun(x, y, C0=C_out):
            F, D, E, C, A, B = C0  # curve parameters (known)

            z = F + D * x + A * x ** 2 + y * (E + C * x + B * y)

            return z

        def dist_fun(P, P0=Pinit, C0=C_out):
            xs, ys = P  # get proposed xy points
            zs = z_fun(xs, ys, C0)  # solve for z
            x0, y0, z0 = P0

            dist = np.sqrt((xs - x0) ** 2 + (ys - y0) ** 2 + (zs - z0) ** 2)

            return dist

        fit = scipy.optimize.least_squares(dist_fun, [10, 10], max_nfev=1e3, ftol=1e-12)
        x, y = fit.x
        z = z_fun(x, y, C_out)
        # x, y, z, _ = fsolve(call_lagrange, np.asarray([100, 100, 100, 1]))

        pca_point_surf[p, :] = x, y, z

    xyz_point_surf = pca.inverse_transform(pca_point_surf)
    # xyz_fin = pca.inverse_transform(pca_fin_full)

    # now, find pd distance for each projected surface point
    pd_pos = np.empty((pca_fin_full.shape[0],))
    pd_i = np.empty((pca_fin_full.shape[0],))

    for p in range(pca_point_surf.shape[0]):
        surf_point = pca_point_surf[p, :]
        dist_list = np.empty((pca_pd_base.shape[0],))
        for pd in range(pca_pd_base.shape[0]):
            dist_list[pd] = calculate_path_distance(FDECAB=C_out, p0=surf_point, p1=pca_pd_base[pd, :])
            # dist_list[pd] = np.sqrt(np.sum((surf_point-pca_pd_base[pd, :])**2))

        # find closest base point
        # print(dist_list)
        min_i = np.argmin(np.abs(dist_list))
        pd_i[p] = min_i
        pd_pos[p] = dist_list[min_i]

    print(pd_i)
    pd_i = pd_i.astype(int)
    ap_pos = ap_pos_base[pd_i] # use ID of closest base point to assign AP position
    f = go.Figure()
    # f.add_trace(go.Scatter3d(mode='markers', x=pca_point_surf[:, 0], y=pca_point_surf[:, 1], z=pca_point_surf[:, 2],
    #                          marker=dict(color=pd_pos, opacity=0.7)))
    #
    f.add_trace(
        go.Scatter3d(mode='markers', x=pca_pd_base[:, 0], y=pca_pd_base[:, 1], z=pca_pd_base[:, 2],
                     marker=dict(color=ap_pos_base, opacity=0.7)))
    # f.show()
    # f = go.Figure()
    # for p in range(0, 40):
    #     f.add_trace(go.Scatter3d(x=[pca_fin_full[p, 0], pca_point_surf[p, 0]],
    #                              y=[pca_fin_full[p, 1], pca_point_surf[p, 1]],
    #                              z=[pca_fin_full[p, 2], pca_point_surf[p, 2]]))
    #
    # f.add_trace(go.Surface(x=P0, y=P1, z=PP2_curve, opacity=0.75, showscale=False))
    f.show()
    return xyz_fit_curve, C_out, surf_cm, P0, xyz_plane, add_array, xyz_pd_array, xyz_point_surf, pd_pos, ap_pos

def cart_to_sphere(xyz):
    ptsnew = np.zeros(xyz.shape)
    xy = xyz[:, 0]**2 + xyz[:, 1]**2
    ptsnew[:, 0] = np.sqrt(xy + xyz[:, 2]**2)
    ptsnew[:, 1] = np.arctan2(np.sqrt(xy), xyz[:, 2]) # for elevation angle defined from Z-axis down
    ptsnew[:, 2] = np.arctan2(xyz[:, 1], xyz[:, 0])
    return ptsnew

def sphereFit(spX,spY,spZ):
    #   Assemble the A matrix
    spX = np.array(spX)
    spY = np.array(spY)
    spZ = np.array(spZ)
    A = np.zeros((len(spX),4))
    A[:,0] = spX*2
    A[:,1] = spY*2
    A[:,2] = spZ*2
    A[:,3] = 1

    #   Assemble the f matrix
    f = np.zeros((len(spX),1))
    f[:,0] = (spX*spX) + (spY*spY) + (spZ*spZ)
    C, residules, rank, singval = np.linalg.lstsq(A,f)

    #   solve for the radius
    t = (C[0]*C[0])+(C[1]*C[1])+(C[2]*C[2])+C[3]
    radius = math.sqrt(t)

    return radius, C[0], C[1], C[2]

def sphereFit_fixed_r(spX,spY,spZ,r0):
    #   Assemble the A matrix
    spX = np.array(spX)
    spY = np.array(spY)
    spZ = np.array(spZ)

    xyz_array = np.zeros((len(spX), 3))
    xyz_array[:, 0] = spX
    xyz_array[:, 1] = spY
    xyz_array[:, 2] = spZ
    c0 = np.mean(xyz_array, axis=0)
    c0[2] = -250
    def ob_fun(c0, xyz=xyz_array, r=r0):
        res = np.sqrt((xyz[:, 0]-c0[0])**2 + (xyz[:, 1]-c0[1])**2 + (xyz[:, 2]-c0[2])**2) - r
        return res

    C = scipy.optimize.least_squares(ob_fun, c0, bounds=([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, 0]))
    #   solve for the radius
    # t = (C[0]*C[0])+(C[1]*C[1])+(C[2]*C[2])+C[3]
    # radius = math.sqrt(t)
    # print(C)
    return r0, C.x[0], C.x[1], C.x[2]

def calculate_distance_metrics(fin_tip_point, point_stat_raw, AG):
    if fin_tip_point:
        cm_vec = [fin_tip_point[0]["x"], fin_tip_point[0]["y"], fin_tip_point[0]["z"]]
    else:
        cm_vec = np.mean(point_stat_raw.iloc[:, 0:3], axis=0)

    xyz_array_cm = point_stat_raw.iloc[:, 0:3].to_numpy() - cm_vec
    # calculate higher-order position stats
    xyz_df = pd.DataFrame(xyz_array_cm, columns=["X", "Y", "Z"])
    sph_arr = cart_to_sphere(xyz_array_cm)
    sph_df = pd.DataFrame(sph_arr, columns=["r", "lat", "lon"])
    for i in product(xyz_df, xyz_df, repeat=1):
        name = "*".join(i)
        xyz_df[name] = xyz_df[list(i)].prod(axis=1)

    # xyz_df_norm = xyz_df.copy() - np.min(xyz_df.iloc[:], axis=0)
    # xyz_df_norm = np.divide(xyz_df_norm, np.max(xyz_df_norm.iloc[:], axis=0))
    # Graph-based stuff
    if fin_tip_point:
        fin_tip_ind = fin_tip_point[0]["pointNumber"]
    else:
        fin_tip_ind = []
    tip_dists = calculate_network_distances(fin_tip_ind, AG)
    # print(sph_df.head())
    # combine
    point_stat_df = pd.concat((sph_df, xyz_df), axis=1)
    point_stat_df["fin_tip_dists"] = tip_dists
    # print(point_stat_df.columns)
    # point_stat_df = point_stat_raw.iloc[:, 0]

    return point_stat_df

def calculate_network_distances(point, G):

    dist_array = np.empty((len(G)))
    if point:
        graph_distances = nx.single_source_dijkstra_path_length(G, str(point))

        for d in range(len(G)):
            try:
                dist_array[d] = graph_distances[str(d)]
            except:
                dist_array[d] = -1

        max_dist = np.max(dist_array)
        dist_array[np.where(dist_array == -1)[0]] = max_dist + 1
    else:
        dist_array[:] = 100

    return dist_array

def calculate_adjacency_graph(df):
    k_nn = 5

    # calculate KD tree and use this to determine k nearest neighbors for each point
    xyz_array = df[["X", "Y", "X"]]
    # print(xyz_array)
    tree = KDTree(xyz_array)

    # get nn distances
    nearest_dist, nearest_ind = tree.query(xyz_array, k=k_nn + 1)

    # find average distance to kth closest neighbor
    mean_nn_dist_vec = np.mean(nearest_dist, axis=0)
    nn_thresh = mean_nn_dist_vec[k_nn]

    # iterate through points and build adjacency network
    G = nx.Graph()

    for n in range(xyz_array.shape[0]):
        node_to = str(n)
        nodes_from = [str(f) for f in nearest_ind[n, 1:]]
        weights_from = [w for w in nearest_dist[n, 1:]]

        # add node
        G.add_node(node_to)
        # only allow edges that are within twice the average k_nn distance
        allowed_edges = np.where(weights_from <= 2 * nn_thresh)[0]
        for a in allowed_edges:
            G.add_edge(node_to, nodes_from[a], weight=weights_from[a])

    return G

def calculate_point_cloud_stats(df):
    ########################
    # convert to point cloud
    xyz_array = np.asarray(df[["X", "Y", "Z"]]).copy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_array)

    # generate features to train classifier
    cloud = PyntCloud.from_instance("open3d", pcd)

    k_vec = [101]

    for k in range(len(k_vec)):
        k_neighbors = cloud.get_neighbors(k=k_vec[k])

        ev = cloud.add_scalar_field("eigen_values", k_neighbors=k_neighbors)
        # cloud.add_scalar_field("eigen_decomposition", k_neighbors=k_neighbors)

        cloud.add_scalar_field("curvature", ev=ev)
        # cloud.add_scalar_field("anisotropy", ev=ev)
        cloud.add_scalar_field("eigenentropy", ev=ev)
        # cloud.add_scalar_field("eigen_sum", ev=ev)
        # cloud.add_scalar_field("linearity", ev=ev)
        # cloud.add_scalar_field("omnivariance", ev=ev)
        cloud.add_scalar_field("planarity", ev=ev)
        cloud.add_scalar_field("sphericity", ev=ev)

    point_stat_raw = cloud.points
    point_stat_raw = point_stat_raw.filter(regex="planar|curv")
    point_stat_raw = pd.concat((cloud.points.iloc[:, 0:3], point_stat_raw), axis=1)
    return point_stat_raw


def logistic_regression(features_train, features_all, fin_class, model=None):

    # print(features_train.head(2))
    if model == None:
        model = RandomForestClassifier(random_state=0)#, class_weight='balanced')

    # features_train = features_train[:, np.newaxis]

    model.fit(features_train, fin_class.ravel())
    predictions = model.predict(features_all)

    return predictions, model
def polyval2d(x, y, m):
    order = int(np.sqrt(len(m))) - 1
    ij = itertools.product(range(order + 1), range(order + 1))
    z = np.zeros_like(x)
    for a, (i, j) in zip(m, ij):
        z += a * x ** i * y ** j
    return z

def polyfit2d(x, y, z, order=2):
    ncols = (order + 1) ** 2
    G = np.zeros((x.size, ncols))
    ij = itertools.product(range(order + 1), range(order + 1))
    for k, (i, j) in enumerate(ij):
        G[:, k] = x ** i * y ** j
    m, _, _, _ = np.linalg.lstsq(G, z)
    return m



def segment_pec_fins(dataRoot):

    global fileList, imNameList
    # get list of filepaths
    fileList = sorted(glob.glob(dataRoot + '*_nucleus_props.csv'))
    imNameList = []
    for fn in range(len(fileList)):
        labelName = fileList[fn].replace(dataRoot, '', 1)
        labelName = labelName.replace('_nucleus_props.csv', '')
        imNameList.append(labelName)

    #filename = imNameList[0]
    #global fin_points_prev, base_points_prev, class_predictions_curr, curationPath, propPath

    #fin_points_prev = []
    #base_points_prev = []

    def load_nucleus_dataset(filename):

        propPath = dataRoot + filename + '_nucleus_props.csv'
        curationPath = dataRoot + filename + '_curation_info/'
        modelPath = dataRoot + 'model.sav'
        modelDataPath = dataRoot + 'model_data.csv'
        if not os.path.isdir(curationPath):
            os.mkdir(curationPath)

        model = []
        if os.path.isfile(modelPath):
            model = pickle.load(open(modelPath, 'rb'))

        model_data = []
        if os.path.isfile(modelDataPath):
            model_data = pd.read_csv(modelDataPath)

        if os.path.isfile(propPath):
            df = pd.read_csv(propPath, index_col=0)
            # clean up columns
            # df = df[df.columns.drop(list(df.filter(regex='Unnamed')))]
            # print(df.head(1))
        else:
            raise Exception(
                f"Selected dataset( {filename} ) dataset has no nucleus data. Have you run extract_nucleus_stats?")

        base_points_prev = []
        fin_points_prev = []
        other_points_prev = []
        class_predictions_curr = []
        fin_tip_prev = []

        # load key info from previous session
        base_path = curationPath + 'base_points.pkl'
        other_path = curationPath + 'other_points.pkl'
        fin_path = curationPath + 'fin_points.pkl'
        fin_tip_path = curationPath + 'fin_tip_point.pkl'

        if os.path.isfile(base_path):
            with open(base_path, 'rb') as fn:
                base_points_prev = pickle.load(fn)
            base_points_prev = json.loads(base_points_prev)

        if os.path.isfile(other_path):
            with open(other_path, 'rb') as fn:
                other_points_prev = pickle.load(fn)
            other_points_prev = json.loads(other_points_prev)

        if os.path.isfile(fin_path):
            with open(fin_path, 'rb') as fn:
                fin_points_prev = pickle.load(fn)
            fin_points_prev = json.loads(fin_points_prev)

        if os.path.isfile(fin_tip_path):
            with open(fin_tip_path, 'rb') as fn:
                fin_tip_prev = pickle.load(fn)
            fin_tip_prev = json.loads(fin_tip_prev)

        if 'pec_fin_flag' in df:
            class_predictions_curr = df['pec_fin_flag']

        return {"df": df, "class_predictions_curr": class_predictions_curr, "fin_points_prev": fin_points_prev,
                "base_points_prev": base_points_prev, "other_points_prev": other_points_prev, "propPath": propPath,
                "fin_tip_prev": fin_tip_prev, "curationPath": curationPath, "model_data": model_data, "model": model}

    df_dict = load_nucleus_dataset(imNameList[0])

    global base_points_prev, fin_points_prev, other_points_prev, fin_tip_prev, class_predictions_curr

    df = df_dict["df"]
    base_points_prev = df_dict["base_points_prev"]
    other_points_prev = df_dict["other_points_prev"]
    fin_points_prev = df_dict["fin_points_prev"]
    fin_tip_prev = df_dict["fin_tip_prev"]
    class_predictions_curr = df_dict["class_predictions_curr"]

    ####
    # set plot boundaries


    ########################
    # App
    app = dash.Dash(__name__)#, external_stylesheets=external_stylesheets)
    global init_toggle
    init_toggle = True
    def create_figure(df):
        # calculate axis limits
        xmin = np.min(df["X"].iloc[:]) + 5
        xmax = np.max(df["X"].iloc[:]) - 5

        ymin = np.min(df["Y"].iloc[:]) - 5
        ymax = np.max(df["Y"].iloc[:]) + 5

        zmin = np.min(df["Z"].iloc[:]) - 3
        zmax = np.max(df["Z"].iloc[:]) + 3

        r_vec = np.sqrt(df["X"]**2 + df["Y"]**2 + df["Z"]**2)
        fig = px.scatter_3d(df, x="X", y="Y", z="Z", opacity=0.4, color=r_vec, color_continuous_scale='ice')
        fig.update_traces(marker=dict(size=6))
        fig.update_layout(coloraxis_showscale=False)
        # fig.update_coloraxes(showscale=False)
        fig.update_layout(
            scene=dict(
                xaxis=dict(range=[xmin, xmax], ),
                yaxis=dict(range=[ymin, ymax], ),
                zaxis=dict(range=[zmin, zmax], ),
                aspectratio=dict(x=1, y=1, z=0.5)))

        return fig

    f = create_figure(df)

    app.layout = html.Div([
                        html.Button('Clear Selection', id='clear'),
                        html.Button('Save', id='save-button'),
                        html.Button('Calculate Fit', id='calc-button'),
                        html.P(id='save-button-hidden', style={'display': 'none'}),
                        dcc.Graph(id='3d_scat', figure=f),
                        html.Div(id='df_list', hidden=True),
                        html.Div(id='other_points', hidden=True),
                        html.Div(id='base_points', hidden=True),
                        html.Div(id='fin_points', hidden=True),
                        html.Div(id='fin_tip_point', hidden=True),
                        html.Div(id='pfin_nuclei', hidden=True),
                      
                        html.Div([
                            dcc.Dropdown(imNameList, imNameList[0], id='dataset-dropdown'),
                            html.Div(id='dd-output-container', hidden=True)
                        ],
                            style={'width': '30%', 'display': 'inline-block'}),
                        html.Div([
                            dcc.Dropdown(["Fin Tip", "Pec Fin", "Base Surface", "Other"], "Fin Tip", id='class-dropdown'),
                            html.Div(id='class-output-container', hidden=True)
                        ],
                            style={'width': '30%', 'display': 'inline-block'})
                        ]
                        )

    @app.callback(
        Output('dd-output-container', 'children'),
        Input('dataset-dropdown', 'value')
    )
    def load_wrapper(value):
        #[xyz_array, base_points_prev, fin_points_prev, class_predictions_curr])
        return value

    @app.callback(
        Output('class-output-container', 'children'),
        Input('class-dropdown', 'value')
    )
    def load_wrapper(value):
        # [xyz_array, base_points_prev, fin_points_prev, class_predictions_curr])
        return value

    @app.callback([Output('base_points', 'children'),
                   Output('other_points', 'children'),
                   Output('fin_points', 'children'),
                   Output('fin_tip_point', 'children')],
                    [Input('3d_scat', 'clickData'),
                     Input('clear', 'n_clicks'),
                     Input('class-output-container', 'children'),
                     Input('dd-output-container', 'children')],
                    [State('base_points', 'children'),
                     State('other_points', 'children'),
                     State('fin_points', 'children'),
                     State('fin_tip_point', 'children')
                     ])

    def select_point(clickData, n_clicks, class_val, fileName, base_points, other_points, fin_points, fin_tip_point):
        ctx = dash.callback_context
        ids = [c['prop_id'] for c in ctx.triggered]
        changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
        # toggle_val = toggle_switch == "Click to select Pec Fin points."

        df_dict = load_nucleus_dataset(fileName)

        base_points_prev = df_dict["base_points_prev"]
        fin_points_prev = df_dict["fin_points_prev"]
        other_points_prev = df_dict["other_points_prev"]
        fin_tip_prev = df_dict["fin_tip_prev"]

        if base_points and ('dd-output-container' not in changed_id):
            base_results = json.loads(base_points)
        else:
            base_results = []

        if other_points and ('dd-output-container' not in changed_id):
            other_results = json.loads(other_points)
        else:
            other_results = []

        if fin_points and ('dd-output-container' not in changed_id):
            fin_results = json.loads(fin_points)
        else:
            fin_results = []

        if fin_tip_point and ('dd-output-container' not in changed_id):
            fin_tip_point = json.loads(fin_tip_point)
        else:
            fin_tip_point = []

        global init_toggle # NL: does this do anything?

        if ('dd-output-container' in changed_id) | init_toggle:
            if len(base_points_prev) > 0:
                for p in base_points_prev:
                    if p not in base_results: # NL: Does this do anything? why would there be base results already?
                        base_results.append(p)

            if len(fin_points_prev) > 0:
                for p in fin_points_prev:
                    if p not in fin_results:
                        fin_results.append(p)

            if len(other_points_prev) > 0:
                for p in other_points_prev:
                    if p not in other_results:
                        other_results.append(p)
            if fin_tip_prev:
                fin_tip_point = [fin_tip_prev[0]]

        if '3d_scat.clickData' in ids:
            if class_val == "Base Surface":
                xyz_nf = np.round([[base_results[i]["x"], base_results[i]["y"], base_results[i]["z"]] for i in
                                   range(len(base_results))], 3)
                for p in clickData['points']:
                    xyz_p = np.round([p["x"], p["y"], p["z"]], 3)
                    if [xyz_p] not in xyz_nf:
                        base_results.append(p)
                    else:
                        # if selected point is in list, then lets unselect
                        rm_ind = np.where(xyz_nf == xyz_p)
                        base_results.pop(rm_ind[0][0])

            elif class_val == "Pec Fin":
                xyz_f = np.round([[fin_results[i]["x"], fin_results[i]["y"], fin_results[i]["z"]] for i in range(len(
                    fin_results))], 3)
                for p in clickData['points']:
                    xyz_p = np.round([p["x"], p["y"], p["z"]], 3)
                    if [xyz_p] not in xyz_f:
                        fin_results.append(p)
                    else:
                        # if selected point is in list, then lets unselect
                        rm_ind = np.where(xyz_f == xyz_p)
                        fin_results.pop(rm_ind[0][0])

            elif class_val == "Other":
                xyz_o = np.round([[other_results[i]["x"], other_results[i]["y"], other_results[i]["z"]] for i in range(len(
                    other_results))], 3)
                for p in clickData['points']:
                    xyz_p = np.round([p["x"], p["y"], p["z"]], 3)
                    if [xyz_p] not in xyz_o:
                        other_results.append(p)
                    else:
                        # if selected point is in list, then lets unselect
                        rm_ind = np.where(xyz_o == xyz_p)
                        other_results.pop(rm_ind[0][0])

            elif class_val == "Fin Tip":
                fin_tip_point = clickData['points']
        if ('clear' in changed_id):
            base_results = []
            fin_results = []
            other_results = []
            fin_tip_point = []

        if fin_tip_point:
            if "x" not in fin_tip_point[0]:
                fin_tip_point = []

        base_results = json.dumps(base_results)
        fin_results = json.dumps(fin_results)
        other_results = json.dumps(other_results)
        fin_tip_point = json.dumps(fin_tip_point)

        init_toggle = False
        return base_results, other_results, fin_results, fin_tip_point

    @app.callback([Output('3d_scat', 'figure'),
                   Output('pfin_nuclei', 'children')],
                [Input('pfin_nuclei', 'children'),
                 Input('base_points', 'children'),
                 Input('other_points', 'children'),
                 Input('fin_points', 'children'),
                 Input('fin_tip_point', 'children'),
                 Input('calc-button', 'n_clicks'),
                 Input('dd-output-container', 'children')])

    def chart_3d(class_predictions_in, base_points, other_points, fin_points, fin_tip_point, n_clicks, fileName):

        global f

        # check to see which values have changed
        changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

        df_dict = load_nucleus_dataset(fileName)
        df = df_dict["df"]

        # calculate axis limits
        xmin = np.min(df["X"].iloc[:]) + 5
        xmax = np.max(df["X"].iloc[:]) - 5

        ymin = np.min(df["Y"].iloc[:]) - 5
        ymax = np.max(df["Y"].iloc[:]) + 5

        zmin = np.min(df["Z"].iloc[:]) - 3
        zmax = np.max(df["Z"].iloc[:]) + 3

        class_predictions_curr = df_dict["class_predictions_curr"]

        f = create_figure(df)

        f.update_layout(uirevision="Don't change")
        base_points = json.loads(base_points) if base_points else []
        fin_points = json.loads(fin_points) if fin_points else []
        other_points = json.loads(other_points) if other_points else []
        fin_tip_point = json.loads(fin_tip_point) if fin_tip_point else []
        if fin_tip_point:
            f.add_trace(
                go.Scatter3d(
                    mode='markers',
                    x=[fin_tip_point[0]["x"]],
                    y=[fin_tip_point[0]["y"]],
                    z=[fin_tip_point[0]["z"]],
                    marker=dict(
                        color='black',
                        size=8,
                        line=dict(
                            color='black',
                            width=4
                        )
                    ),
                    showlegend=False
                )
            )
        if base_points:
            f.add_trace(
                go.Scatter3d(
                    mode='markers',
                    x=[p['x'] for p in base_points],
                    y=[p['y'] for p in base_points],
                    z=[p['z'] for p in base_points],
                    marker=dict(
                        color='crimson',
                        size=7,
                        line=dict(
                            color='black',
                            width=4
                        )
                    ),
                    showlegend=False
                )
            )

        if fin_points:
            f.add_trace(
                go.Scatter3d(
                    mode='markers',
                    x=[p['x'] for p in fin_points],
                    y=[p['y'] for p in fin_points],
                    z=[p['z'] for p in fin_points],
                    marker=dict(
                        color='lightgreen',
                        size=7,
                        line=dict(
                            color='black',
                            width=4
                        )
                    ),
                    showlegend=False
                )
            )

        if other_points:
            f.add_trace(
                go.Scatter3d(
                    mode='markers',
                    x=[p['x'] for p in other_points],
                    y=[p['y'] for p in other_points],
                    z=[p['z'] for p in other_points],
                    marker=dict(
                        color='azure',
                        size=7,
                        line=dict(
                            color='black',
                            width=4
                        )
                    ),
                    showlegend=False
                )
            )

        # global class_predictions_curr
        if 'calc-button' in changed_id:
            if base_points and fin_points and other_points:

                # match clicked points to ref points using xyz coordinates
                xyz_click = np.empty((len(other_points) + len(base_points) + len(fin_points), 3))
                for o, op in enumerate(other_points):
                    xyz_click[o, :] = np.round([op["x"], op["y"], op["z"]], 3)

                for b, bp in enumerate(base_points):
                    xyz_click[b+len(other_points), :] = np.round([bp["x"], bp["y"], bp["z"]], 3)

                for fn, fp in enumerate(fin_points):
                    xyz_click[fn+len(other_points)+len(base_points), :] = np.round([fp["x"], fp["y"], fp["z"]], 3)

                dist_mat = distance_matrix(xyz_click, np.asarray(df[["X", "Y", "Z"]]))
                lb_indices = np.argmin(dist_mat, axis=1)

                # calculate point cloud stats
                point_stat_raw = calculate_point_cloud_stats(df)

                ################
                # calculate network-based stats
                G = calculate_adjacency_graph(df)

                point_stat_df = calculate_distance_metrics(fin_tip_point, point_stat_raw, G)

                # generate class vec
                fin_class_vec = np.zeros((len(base_points) + len(other_points) + len(fin_points), 1))
                fin_class_vec[len(other_points):, 0] = 1  # base=1
                fin_class_vec[len(base_points) + len(other_points):, 0] = 2  # fin=2

                df_lb = point_stat_df.iloc[lb_indices]

                class_predictions, model = logistic_regression(df_lb, point_stat_df, fin_class_vec)

            else:
                class_predictions = np.zeros((df.shape[0],))

        elif (class_predictions_in != None) and ('dd-output-container' not in changed_id):
            class_predictions = class_predictions_in
        else:
            class_predictions = class_predictions_curr

        pec_fin_nuclei = np.where(np.asarray(class_predictions) == 2)[0]
        base_nuclei = np.where(np.asarray(class_predictions) == 1)[0]
        other_nuclei = np.where(np.asarray(class_predictions) == 0)[0]

        if (len(base_nuclei) > 0) and (len(pec_fin_nuclei) > 0):
            # fit sphere
            xyz_base = df[["X", "Y", "Z"]].iloc[base_nuclei]
            xyz_base = xyz_base.to_numpy()
            # print(xyz_base)
            r, x0, y0, z0 = sphereFit_fixed_r(xyz_base[:, 0], xyz_base[:, 1], xyz_base[:, 2], 250)
            u, v = np.mgrid[0:2 * np.pi:200j, 0:np.pi:100j]
            x = np.cos(u) * np.sin(v) * r
            y = np.sin(u) * np.sin(v) * r
            z = np.cos(v) * r
            x_sphere = x + x0
            y_sphere = y + y0
            z_sphere = z + z0

            # add sphere to plot
            f.add_trace(go.Surface(x=x_sphere, y=y_sphere, z=z_sphere, opacity=0.5, showscale=False))

            c0 = np.asarray([x0, y0, z0])

            xyz_fit_curve, surf_params, surf_cm, P0, xyz_plane, add_array, xyz_pd_array, xyz_point_surf, pd_pos_vec, ap_pos_vec \
                                                                      = fit_quadratic_surface(df, c0, pec_fin_nuclei, r)

            
            X_fit = np.reshape(xyz_fit_curve[:, 0], (P0.shape))
            Y_fit = np.reshape(xyz_fit_curve[:, 1], (P0.shape))
            Z_fit_curve = np.reshape(xyz_fit_curve[:, 2], (P0.shape))

            X_fit_plane = np.reshape(xyz_plane[:, 0], (P0.shape))
            Y_fit_plane = np.reshape(xyz_plane[:, 1], (P0.shape))
            Z_fit_plane = np.reshape(xyz_plane[:, 2], (P0.shape))
            
            ##########
            # filter for points inside the fin
            # keep only points that are inside the fin
            xyz_fin = df[["X", "Y", "Z"]].iloc[pec_fin_nuclei].to_numpy()
            xyz_array_norm = np.divide(xyz_fin, np.asarray([np.max(xyz_fin[:, 0]),
                                                                 np.max(xyz_fin[:, 1]),
                                                                 np.max(xyz_fin[:, 2])]))
            # generate normalized arrays
            X_norm = X_fit / np.max(xyz_fin[:, 0])
            Y_norm = Y_fit / np.max(xyz_fin[:, 1])
            Z_norm = Z_fit_curve / np.max(xyz_fin[:, 2])

            alpha_fin = alphashape.alphashape(xyz_array_norm, 4)

            xyz_long = np.concatenate((np.reshape(X_norm, (X_norm.size, 1)),
                                       np.reshape(Y_norm, (X_norm.size, 1)),
                                       np.reshape(Z_norm, (X_norm.size, 1))),
                                      axis=1)

            inside_flags = alpha_fin.contains(xyz_long)
            inside_mat = np.reshape(inside_flags, (P0.shape))

            # Z_fit_curve_filt = Z_fit_curve
            # Z_fit_curve_filt[np.where(~inside_mat)] = np.nan

            # Find base points to use as PD references
            sp_dist_array = np.abs(np.sqrt(np.sum((xyz_pd_array - c0)**2, axis=1)) - r)
            # print(xyz_pd_array)
            close_indices = np.where(sp_dist_array <= 1)[0]
            pd_base_xyz = xyz_pd_array[close_indices]

            # f = go.Figure()
            # f.add_trace(go.Surface(x=X_fit, y=Y_fit, z=Z_fit_curve, opacity=0.75, showscale=False))

            # find mid points to use as AP references

            #
            # f.add_trace(go.Surface(x=X_fit, y=Y_fit, z=Z_fit_curve_filt, opacity=0.75, showscale=False))
            #
            # f.add_trace(go.Mesh3d(x=xyz_fin[:, 0], y=xyz_fin[:, 1], z=xyz_fin[:, 2],
            #                       alphahull=9,
            #                       opacity=0.5,
            #                       color='gray'))
            #
            # f.add_trace(go.Surface(x=X_fit_plane, y=Y_fit_plane, z=Z_fit_plane, opacity=0.75, showscale=False))
            # for p in range(700, 740):
            #     f.add_trace(go.Scatter3d(x=[xyz_fin[p, 0], xyz_point_surf[p, 0]],
            #                              y=[xyz_fin[p, 1], xyz_point_surf[p, 1]],
            #                              z=[xyz_fin[p, 2], xyz_point_surf[p, 2]]))

                # f.add_trace(go.Scatter3d(mode='markers', x=[xyz_fin[p, 0]],
                #                          y=[xyz_fin[p, 1]],
                #                          z=[xyz_fin[p, 2]]))

            #
            f.add_trace(go.Scatter3d(mode='markers', x=xyz_point_surf[:, 0], y=xyz_point_surf[:, 1], z=xyz_point_surf[:, 2],
                                     marker=dict(color=ap_pos_vec, opacity=0.7)))

            f.add_trace(
                go.Scatter3d(mode='markers', x=pd_base_xyz[:, 0], y=pd_base_xyz[:, 1], z=pd_base_xyz[:, 2],
                             marker=dict(opacity=0.7)))


            #                          mode='markers', opacity=0.6))


        ################
        # Plot predictions
        # f.add_trace(
        #     go.Scatter3d(
        #         mode='markers',
        #         x=[df["X"].iloc[p] for p in pec_fin_nuclei],
        #         y=[df["Y"].iloc[p] for p in pec_fin_nuclei],
        #         z=[df["Z"].iloc[p] for p in pec_fin_nuclei],
        #         marker=dict(
        #             color='lightgreen',
        #             opacity=0.1,
        #             size=5),
        #         showlegend=False
        #     )
        # )
        #
        # f.add_trace(
        #     go.Scatter3d(
        #         mode='markers',
        #         x=[df["X"].iloc[p] for p in base_nuclei],
        #         y=[df["Y"].iloc[p] for p in base_nuclei],
        #         z=[df["Z"].iloc[p] for p in base_nuclei],
        #         marker=dict(
        #             color='coral',
        #             opacity=0.1,
        #             size=5),
        #         showlegend=False
        #     )
        # )
        #
        # f.add_trace(
        #     go.Scatter3d(
        #         mode='markers',
        #         x=[df["X"].iloc[p] for p in other_nuclei],
        #         y=[df["Y"].iloc[p] for p in other_nuclei],
        #         z=[df["Z"].iloc[p] for p in other_nuclei],
        #         marker=dict(
        #             color='azure',
        #             opacity=0.1,
        #             size=5),
        #         showlegend=False
        #     )
        # )


        return f, class_predictions

    @app.callback(
        Output('save-button-hidden', 'children'),
        [Input('save-button', 'n_clicks'),
            Input('pfin_nuclei', 'children'),
            Input('base_points', 'children'),
            Input('fin_points', 'children'),
            Input('other_points', 'children'),
            Input('fin_tip_point', 'children'),
            Input('dd-output-container', 'children')])

    def clicks(n_clicks, class_predictions, base_points, fin_points, other_points, fin_tip_point, fileName):
        changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

        if 'save-button' in changed_id:
            # save
            df_dict = load_nucleus_dataset(fileName)
            df = df_dict["df"]
            class_predictions = class_predictions
            if len(class_predictions) == df.shape[0]:
                df["pec_fin_flag"] = class_predictions
            write_file = dataRoot + fileName + '_nucleus_props.csv'
            df.to_csv(write_file)

            write_file2 = dataRoot + fileName + '_curation_info/base_points.pkl'
            with open(write_file2, 'wb') as wf:
                pickle.dump(base_points, wf)

            write_file3 = dataRoot + fileName + '_curation_info/fin_points.pkl'
            with open(write_file3, 'wb') as wf:
                pickle.dump(fin_points, wf)

            write_file4 = dataRoot + fileName + '_curation_info/other_points.pkl'
            with open(write_file4, 'wb') as wf:
                pickle.dump(other_points, wf)

            write_file5 = dataRoot + fileName + '_curation_info/fin_tip_point.pkl'
            with open(write_file5, 'wb') as wf:
                pickle.dump(fin_tip_point, wf)

            # save
            # model_file = dataRoot + 'model.sav'
            # pickle.dump(model, open(model_file, 'wb'))


    app.run_server(debug=True, port=8053)

if __name__ == '__main__':

    # set parameters
    filename = "2022_12_15 HCR Hand2 Tbx5a Fgf10a_1.zarr"
    dataRoot = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/pecFin/HCR_Data/nucleus_props/"
    level = 1

    # load image data
    segment_pec_fins(dataRoot)

    #nucleus_coordinates = pd.read_csv('/Users/nick/test.csv')
    #print(nucleus_coordinates)
