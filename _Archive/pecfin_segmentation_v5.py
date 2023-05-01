import dash
import plotly.express as px
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import json
import glob2 as glob
from skimage.measure import regionprops
import itertools
from scipy.spatial import distance_matrix
from sklearn.neighbors import KDTree
from sklearn.cluster import DBSCAN
import pickle
import os.path
import os
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import dash_daq as daq
from itertools import product
import math

# define a function to fit a sphere to points
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

def logistic_regression(xyz_train, xyz_all, fin_class, RF_flag=True, reps=2):

    # convert to pandas data frames
    xyz_train_df = pd.DataFrame(xyz_train.astype(float), columns=["X", "Y", "Z"])
    xyz_all_df = pd.DataFrame(xyz_all.astype(float), columns=["X", "Y", "Z"])

    # expand
    xyz_train_exp = pd.DataFrame()
    for i in product(xyz_train_df, xyz_train_df, repeat=reps):
        name = "*".join(i)
        xyz_train_exp[name] = xyz_train_df[list(i)].prod(axis=1)

    xyz_all_exp = pd.DataFrame()
    for i in product(xyz_all_df, xyz_all_df, repeat=reps):
        name = "*".join(i)
        xyz_all_exp[name] = xyz_all_df[list(i)].prod(axis=1)

    if not RF_flag:
        # perform logistic regression
        logistic_reg = LogisticRegression()
        logistic_reg.fit(xyz_train_exp, fin_class.ravel())
        predictions = logistic_reg.predict(xyz_all_exp)
    else:
        clf = RandomForestClassifier(max_depth=2, random_state=0)
        clf.fit(xyz_train_exp, fin_class.ravel())
        predictions = clf.predict(xyz_all_exp)

    return predictions
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

def visualize_atlas(dataRoot):

    global fileList, imNameList
    # get list of filepaths
    fileList = sorted(glob.glob(dataRoot + '*_nucleus_props.csv'))
    imNameList = []
    for fn in range(len(fileList)):
        labelName = fileList[fn].replace(dataRoot, '', 1)
        labelName = labelName.replace('_nucleus_props.csv', '')
        imNameList.append(labelName)

    #filename = imNameList[0]
    #global fin_points_prev, not_fin_points_prev, class_predictions_curr, curationPath, propPath

    #fin_points_prev = []
    #not_fin_points_prev = []

    def load_nucleus_dataset(filename):
        #global fin_points_prev, not_fin_points_prev, class_predictions_curr, df, curationPath, propPath

        propPath = dataRoot + filename + '_nucleus_props.csv'
        curationPath = dataRoot + filename + '_curation_info/'

        if not os.path.isdir(curationPath):
            os.mkdir(curationPath)

        if os.path.isfile(propPath):
            df = pd.read_csv(propPath)
        else:
            raise Exception(
                f"Selected dataset( {filename} ) dataset has no nucleus data. Have you run extract_nucleus_stats?")

        not_fin_points_prev = []
        fin_points_prev = []
        class_predictions_curr = []

        # load key info from previous session
        not_fin_path = curationPath + 'not_fin_points.pkl'
        fin_path = curationPath + 'fin_points.pkl'
        if os.path.isfile(not_fin_path):
            with open(not_fin_path, 'rb') as fn:
                not_fin_points_prev = pickle.load(fn)
            not_fin_points_prev = json.loads(not_fin_points_prev)

            with open(fin_path, 'rb') as fn:
                fin_points_prev = pickle.load(fn)
            fin_points_prev = json.loads(fin_points_prev)

        if 'pec_fin_flag' in df:
            class_predictions_curr = df['pec_fin_flag']

        return {"df": df, "class_predictions_curr": class_predictions_curr, "fin_points_prev": fin_points_prev,
                "not_fin_points_prev": not_fin_points_prev, "propPath": propPath, "curationPath": curationPath}

    df_dict = load_nucleus_dataset(imNameList[0])

    global df, not_fin_points_prev, fin_points_prev, class_predictions_curr
    df = df_dict["df"]
    not_fin_points_prev = df_dict["not_fin_points_prev"]
    fin_points_prev = df_dict["fin_points_prev"]
    class_predictions_curr = df_dict["class_predictions_curr"]
    # look for saved data from previous curation effort

    # calculate axis limits based off of data
    global xmin, xmax, ymin, ymax, zmin, zmax

    xmin = 0.75 * np.min(df["X"].iloc[:])
    xmax = 1.25 * np.max(df["X"].iloc[:])

    ymin = 0.75 * np.min(df["Y"].iloc[:])
    ymax = 1.25 * np.max(df["Y"].iloc[:])

    zmin = 0.75 * np.min(df["Z"].iloc[:])
    zmax = 1.25 * np.max(df["Z"].iloc[:])

    # calculate NN statistics
    global nn_thresh
    tree = KDTree(df.iloc[:, 0:2], leaf_size=2)
    nearest_dist, nearest_ind = tree.query(df.iloc[:, 0:2], k=3)
    mean_nn_dist_vec = np.mean(nearest_dist[:, 1:], axis=0)
    nn_thresh = mean_nn_dist_vec[0]

    ########################
    # App
    app = dash.Dash(__name__)#, external_stylesheets=external_stylesheets)
    global init_toggle
    init_toggle = True
    def create_figure(df):
        r_vec = np.sqrt(df["X"]**2 + df["Y"]**2 + df["Z"]**2)
        fig = px.scatter_3d(df, x="X", y="Y", z="Z", opacity=0.4, color=r_vec, color_continuous_scale='ice')
        fig.update_traces(marker=dict(size=6))
        fig.update_layout(coloraxis_showscale=False)

        # fig.update_layout(
        #     scene=dict(
        #         xaxis=dict(range=[xmin, xmax], ),
        #         yaxis=dict(range=[ymin, ymax], ),
        #         zaxis=dict(range=[zmin, zmax], ), ))

        return fig

    #df = pd.DataFrame(columns=["X", "Y", "Z"])
    f = create_figure(df)

    app.layout = html.Div([#html.Button('Delete', id='delete'),
                        html.Button('Clear Selection', id='clear'),
                        html.Button('Save', id='save-button'),
                        html.Button('Calculate Fit', id='calc-button'),
                        html.Button('Fit Sphere', id='sphere-button'),
                        html.P(id='save-button-hidden', style={'display': 'none'}),
                        dcc.Graph(id='3d_scat', figure=f),
                        #html.Div('selected:'),
                        html.Div(id='df_list', hidden=True),
                        html.Div(id='not_fin_points', hidden=True),
                        html.Div(id='fin_points', hidden=True),
                        html.Div(id='pfin_nuclei', hidden=True),
                        html.Div(
                            daq.ToggleSwitch(
                                id='class-toggle-switch',
                                value=False,
                                color='lightgreen'
                            )),
                        html.Div(id='my-toggle-switch-output'),
                        html.Div([
                            dcc.Dropdown(imNameList, imNameList[0], id='dataset-dropdown'),
                            html.Div(id='dd-output-container', hidden=True)
                        ],
                        style={'width': '30%', 'display': 'inline-block'})
                        ]
                        )

    @app.callback(
        Output('my-toggle-switch-output', 'children'),
        Input('class-toggle-switch', 'value'))
    def update_output(value):
        value = json.dumps(value)
        toggle_val = value == "true"
        if toggle_val:
            toggle_string = "Pec Fin"
        else:
            toggle_string = "Non-Pec Fin"
        return f'Click to select {toggle_string} points.'

    @app.callback(
        Output('dd-output-container', 'children'),
        Input('dataset-dropdown', 'value')
    )
    def load_wrapper(value):
        #[xyz_array, not_fin_points_prev, fin_points_prev, class_predictions_curr])
        return value

    @app.callback([Output('not_fin_points', 'children'),
                   Output('fin_points', 'children')],
                    [Input('3d_scat', 'clickData'),
                     Input('clear', 'n_clicks'),
                     Input('my-toggle-switch-output', 'children'),
                     Input('dd-output-container', 'children')],
                    [State('not_fin_points', 'children'),
                     State('fin_points', 'children'),
                     ])

    def select_point(clickData, n_clicks, toggle_switch, fileName, not_fin_points, fin_points):
        ctx = dash.callback_context
        ids = [c['prop_id'] for c in ctx.triggered]
        changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
        toggle_val = toggle_switch == "Click to select Pec Fin points."

        df_dict = load_nucleus_dataset(fileName)

        not_fin_points_prev = df_dict["not_fin_points_prev"]
        fin_points_prev = df_dict["fin_points_prev"]

        if not_fin_points and ('dd-output-container' not in changed_id):
            not_fin_results = json.loads(not_fin_points)
        else:
            not_fin_results = []

        if fin_points and ('dd-output-container' not in changed_id):
            fin_results = json.loads(fin_points)
        else:
            fin_results = []

        # check for saved points
        #global not_fin_points_prev, fin_points_prev
        # not_fin_points_prev = df_list[1]
        global init_toggle

        if ('dd-output-container' in changed_id) | init_toggle:
            if len(not_fin_points_prev) > 0:
                for p in not_fin_points_prev:
                    if p not in not_fin_results:
                        not_fin_results.append(p)

            if len(fin_points_prev) > 0:
                for p in fin_points_prev:
                    if p not in fin_results:
                        fin_results.append(p)


        if '3d_scat.clickData' in ids:
            if not toggle_val:
                xyz_nf = np.round([[not_fin_results[i]["x"], not_fin_results[i]["y"], not_fin_results[i]["z"]] for i in
                                   range(len(not_fin_results))], 3)
                for p in clickData['points']:
                    xyz_p = np.round([p["x"], p["y"], p["z"]], 3)
                    if xyz_p not in xyz_nf:
                        not_fin_results.append(p)
                    else:
                        # if selected point is in list, then lets unselect
                        rm_ind = np.where(xyz_nf == xyz_p)
                        not_fin_results.pop(rm_ind[0][0])
            else:
                xyz_f = np.round([[fin_results[i]["x"], fin_results[i]["y"], fin_results[i]["z"]] for i in range(len(
                    fin_results))], 3)
                for p in clickData['points']:
                    xyz_p = np.round([p["x"], p["y"], p["z"]], 3)
                    if xyz_p not in xyz_f:
                        fin_results.append(p)
                    else:
                        # if selected point is in list, then lets unselect
                        rm_ind = np.where(xyz_f == xyz_p)
                        fin_results.pop(rm_ind[0][0])

        if ('clear' in changed_id):
            not_fin_results = []
            fin_results = []

        not_fin_results = json.dumps(not_fin_results)
        fin_results = json.dumps(fin_results)

        init_toggle = False
        return not_fin_results, fin_results

    @app.callback([Output('3d_scat', 'figure'),
                   Output('pfin_nuclei', 'children')],
                [Input('pfin_nuclei', 'children'),
                 Input('not_fin_points', 'children'),
                 Input('fin_points', 'children'),
                 Input('calc-button', 'n_clicks'),
                 Input('sphere-button', 'n_clicks'),
                 Input('dd-output-container', 'children')])

    def chart_3d(class_predictions_in, not_fin_points, fin_points, n_clicks, n_clicks_sphere, fileName):

        global f

        # check to see which values have changed
        changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

        df_dict = load_nucleus_dataset(fileName)
        df = df_dict["df"]

        class_predictions_curr = df_dict["class_predictions_curr"]

        f = create_figure(df)
        # set plot preferences
        f.update_layout(uirevision="Don't change")

        not_fin_points = json.loads(not_fin_points) if not_fin_points else []
        fin_points = json.loads(fin_points) if fin_points else []

        if not_fin_points:
            f.add_trace(
                go.Scatter3d(
                    mode='markers',
                    x=[p['x'] for p in not_fin_points],
                    y=[p['y'] for p in not_fin_points],
                    z=[p['z'] for p in not_fin_points],
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

        # fit sphere to data if button pressed
        surf_indices = []
        other_indices = []
        labels = []
        if 'sphere-button' in changed_id:
            if len(not_fin_points) >= 3:
                xyz = np.reshape([[p['x'], p['y'], p['z']] for p in not_fin_points], (len(not_fin_points), 3))

                r, x0, y0, z0 = sphereFit(xyz[:, 0], xyz[:, 1], xyz[:, 2])
                u, v = np.mgrid[0:2 * np.pi:200j, 0:np.pi:100j]
                x = np.cos(u) * np.sin(v) * r
                y = np.sin(u) * np.sin(v) * r
                z = np.cos(v) * r
                x = x + x0
                y = y + y0
                z = z + z0

                # add sphere to plot
                f.add_trace(go.Surface(x=x, y=y, z=z, opacity=0.2))

                # hold axes fixed
                f.update_layout(
                    scene=dict(
                        xaxis=dict(range=[xmin, xmax], ),
                        yaxis=dict(range=[ymin, ymax], ),
                        zaxis=dict(range=[zmin, zmax], ), ))

                # use sphere to define surface points and remove them from the set (for now)
                sphere_surf_dist = np.sqrt((df["X"].iloc[:] - x0)**2 +
                                           (df["Y"].iloc[:] - y0)**2 +
                                           (df["Z"].iloc[:] - z0)**2) - r
                surf_indices = np.where(sphere_surf_dist <= 6*nn_thresh)[0] # note that this will include all points within sphere boundary

                # remove surf indices from consideration
                xyz_array = df[["X", "Y", "Z"]].to_numpy()
                other_indices = np.asarray([o for o in range(xyz_array.shape[0]) if o not in surf_indices])
                # print(other_indices)
                xyz_no_surf = xyz_array[other_indices]

                # use density-based clustering to divide remaining points into groups
                clustering = DBSCAN(eps=7*nn_thresh, min_samples=2).fit(xyz_no_surf)
                labels = clustering.labels_
                # add_indices = [s for s in surf_indices if s not in not_fin_points]
                # not_fin_points.append(add_indices)
                # print(clustering)
                # print(np.unique(labels))
            else:
                Warning(f"Not enough surface points selected (need at least 3)")
                surf_indices = []

        # predict fin nuclei if button pressed
        if 'calc-button' in changed_id:
            if not_fin_points and fin_points:
                #oint_dict = json.loads(not_fin_points)
                xyz = np.vstack((
                    np.reshape([[p['x'], p['y'], p['z']] for p in not_fin_points], (len(not_fin_points), 3)),
                    np.reshape([[p['x'], p['y'], p['z']] for p in fin_points], (len(fin_points), 3))
                ))
                fin_class_vec = np.zeros((len(not_fin_points) + len(fin_points), 1))
                fin_class_vec[len(not_fin_points):, 0] = 1

                class_predictions = logistic_regression(xyz, df[["X", "Y", "Z"]], fin_class_vec)

            else:
                class_predictions = np.zeros((df.shape[0],))

        elif (class_predictions_in != None) and ('dd-output-container' not in changed_id):
            class_predictions = class_predictions_in
        else:
            class_predictions = class_predictions_curr

        pec_fin_nuclei = np.where(np.asarray(class_predictions) == 1)[0]
        # f.add_trace(
        #     go.Scatter3d(
        #         mode='markers',
        #         x=[df["X"].iloc[p] for p in pec_fin_nuclei],
        #         y=[df["Y"].iloc[p] for p in pec_fin_nuclei],
        #         z=[df["Z"].iloc[p] for p in pec_fin_nuclei],
        #         marker=dict(
        #             color='lightgreen',
        #             opacity=0.5,
        #             size=5),
        #         showlegend=False
        #     )
        # )
        f.add_trace(
            go.Scatter3d(
                mode='markers',
                x=[df["X"].iloc[p] for p in other_indices],
                y=[df["Y"].iloc[p] for p in other_indices],
                z=[df["Z"].iloc[p] for p in other_indices],

                marker=dict(
                    opacity=1,
                    size=5,
                    color=labels,
                    colorscale="plotly3_r"),
                showlegend=False
            )
        )

        # f.add_trace(
        #     go.Scatter3d(
        #         mode='markers',
        #         x=[df["X"].iloc[p] for p in surf_indices],
        #         y=[df["Y"].iloc[p] for p in surf_indices],
        #         z=[df["Z"].iloc[p] for p in surf_indices],
        #         marker=dict(
        #             color='salmon',
        #             opacity=0.5,
        #             size=5),
        #         showlegend=False
        #     )
        # )

        return f, class_predictions

    @app.callback(
        Output('save-button-hidden', 'children'),
        [Input('save-button', 'n_clicks'),
            Input('pfin_nuclei', 'children'),
            Input('not_fin_points', 'children'),
            Input('fin_points', 'children'),
            Input('dd-output-container', 'children')])

    def clicks(n_clicks, class_predictions, not_fin_points, fin_points, fileName):
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

            write_file2 = dataRoot + fileName + '_curation_info/not_fin_points.pkl'
            with open(write_file2, 'wb') as wf:
                pickle.dump(not_fin_points, wf)

            write_file3 = dataRoot + fileName + '_curation_info/fin_points.pkl'
            with open(write_file3, 'wb') as wf:
                pickle.dump(fin_points, wf)

            # # update and save nucleus dataset
            # fin_class_vec = np.zeros((df.shape[0], 1))
            # fin_class_vec[pec_fin_nuclei] = 1
            # df["pec_fin_flag"] = fin_class_vec
            # df.to_csv(propPath)

            # write_file3 = dataRoot + filename + '_curation_info/value_slider.pkl'
            # with open(write_file3, 'wb') as wf:
            #     pickle.dump(slider_values, wf)

    app.run_server(debug=True, port=8051)

if __name__ == '__main__':

    # set parameters
    filename = "2022_12_15 HCR Hand2 Tbx5a Fgf10a_1.zarr"
    dataRoot = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/pecFin/HCR_Data/nucleus_props/"
    level = 1

    # load image data
    visualize_atlas(dataRoot)

    #nucleus_coordinates = pd.read_csv('/Users/nick/test.csv')
    #print(nucleus_coordinates)
