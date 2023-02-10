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
from skimage.measure import regionprops
import itertools
from scipy.spatial import distance_matrix
from sklearn.neighbors import KDTree
import pickle
import os.path
import os
import math
import dash_daq as daq

def segment_pec_fins(dataRoot, nn_k=10):

    filename = "2022_12_15 HCR Hand2 Tbx5a Fgf10a_1"

    global propPath, curationPath
    global nn_threshold, df

    propPath = dataRoot + filename + '_nucleus_props.csv'
    curationPath = dataRoot + filename + '_curation_info/'

    if not os.path.isdir(curationPath):
        os.mkdir(curationPath)

    if os.path.isfile(propPath):
        df = pd.read_csv(propPath)
    else:
        raise Exception("Selected dataset has no nucleus data. Have you run extract_nucleus_stats?")

    # look for saved data from previous curation effort
    not_fin_points_prev = []
    fin_points_prev = []
    vl_prev = []

    if os.path.isfile(propPath):
        df = pd.read_csv(propPath)

        # load key info from previous session
        sc_path = curationPath + 'not_fin_points.pkl'
        if os.path.isfile(sc_path):
            with open(sc_path, 'rb') as fn:
                not_fin_points_prev = pickle.load(fn)
            not_fin_points_prev = json.loads(not_fin_points_prev)

            vl_path = curationPath + 'value_slider.pkl'
            with open(vl_path, 'rb') as fn:
                vl_prev = pickle.load(fn)

    # calculate NN statistics
    tree = KDTree(df.iloc[:, 0:2], leaf_size=2)
    nearest_dist, nearest_ind = tree.query(df.iloc[:, 0:2], k=nn_k+1)
    mean_nn_dist_vec = np.mean(nearest_dist[:, 1:], axis=0)
    nn_threshold = mean_nn_dist_vec[nn_k-1]

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

    # define mesh grid for curve fitting
    nx = 100
    ny = 100

    global xx, yy
    xm = np.max(df["X"])
    xx, yy = np.meshgrid(np.linspace(0, xm, nx),
                         np.linspace(0, np.max(df["Y"]), ny))

    global val_defaults
    if len(vl_prev) == 2:
        val_defaults = vl_prev
    else:
        val_defaults = [-round(5*nn_threshold), round(nn_threshold)]

    ########################
    # App

    app = dash.Dash(__name__)#, external_stylesheets=external_stylesheets)
    #app.css.append_css({
    #    "external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"
    #})


    def create_figure():
        fig = px.scatter_3d(df, x="X", y="Y", z="Z", opacity=0.5)
        fig.update_traces(marker=dict(size=6))
        return fig
    f = create_figure()

    app.layout = html.Div([#html.Button('Delete', id='delete'),
                        html.Button('Clear Selection', id='clear'),
                        html.Button('Save', id='save-button'),
                        html.Button('Calculate Fit', id='calc-button'),
                        html.P(id='save-button-hidden', style={'display': 'none'}),
                        dcc.Graph(id='3d_scat', figure=f),
                        #html.Div('selected:'),
                        html.Div(id='not_fin_points', hidden=True),
                        html.Div(id='fin_points', hidden=True),
                        html.Div(id='pfin_nuclei', hidden=True),
                        dcc.RangeSlider(-round(10*nn_threshold), round(10*nn_threshold), value=val_defaults, id='select-slider'),
                        html.Div(id='output-container-range-slider'),
                        html.Div(
                            daq.ToggleSwitch(
                                id='class-toggle-switch',
                                value=False
                            )),
                        html.Div(id='my-toggle-switch-output')]
                        )

    @app.callback(
        Output('my-toggle-switch-output', 'children'),
        Input('class-toggle-switch', 'value'))
    def update_output(value):
        value = json.dumps(value)
        return value

    @app.callback([Output('not_fin_points', 'children'),
                   Output('fin_points', 'children')],
                    [Input('3d_scat', 'clickData'),
                     Input('clear', 'n_clicks'),
                     Input('my-toggle-switch-output', 'children')],
                    [State('not_fin_points', 'children'),
                     State('fin_points', 'children')])

    def select_point(clickData, n_clicks, toggle_switch, not_fin_points, fin_points):
        ctx = dash.callback_context
        ids = [c['prop_id'] for c in ctx.triggered]
        changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
        toggle_val = toggle_switch == "false"
        #print(toggle_val)
        # check for previous selections
        if not_fin_points:
            not_fin_results = json.loads(not_fin_points)
        else:
            not_fin_results = []

        if fin_points:
            fin_results = json.loads(fin_points)
        else:
            fin_results = []

        # check for saved points
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
                for p in clickData['points']:
                    if p not in not_fin_results:
                        not_fin_results.append(p)
                    else:
                        # if selected point is in list, then lets unselect
                        not_fin_results.remove(p)
            else:
                for p in clickData['points']:
                    if p not in not_fin_results:
                        fin_results.append(p)
                    else:
                        # if selected point is in list, then lets unselect
                        fin_results.remove(p)

        if 'clear' in changed_id:
            not_fin_results = []
            fin_results = []

        not_fin_results = json.dumps(not_fin_results)
        fin_results = json.dumps(fin_results)
        #coordinates = [[p['x']] for p in not_fin_results]

        return not_fin_results, fin_results

    @app.callback(
        Output('output-container-range-slider', 'children'),
        Input('select-slider', 'value'))
    def update_output(value):
        value = json.dumps(value)
        return value

    @app.callback([Output('3d_scat', 'figure'),
                   Output('pfin_nuclei', 'children')],
                [Input('not_fin_points', 'children'),
                 Input('fin_points', 'children'),
                 Input('output-container-range-slider', 'children')])

    def chart_3d(not_fin_points, fin_points, select_range):
        global f
        # check to see which values have changed
        changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

        f = create_figure()
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
        pec_fin_nuclei = []
        print(fin_points)
        if not_fin_points and (('select-slider' in changed_id) | ('calc-button' in changed_id)):

            #oint_dict = json.loads(not_fin_points)
            xyz = np.reshape([[p['x'], p['y'], p['z']] for p in not_fin_points], (len(not_fin_points), 3))
            # Fit a 3rd order, 2d polynomial
            m = polyfit2d(xyz[:, 0], xyz[:, 1], xyz[:, 2], order=2)
            # Evaluate it on a grid...
            zz = polyval2d(xx, yy, m, )
            # add to figure
            f.add_trace(go.Surface(z=zz, x=xx, y=yy, colorscale='deep', opacity=0.75, showscale=False))

            # now calculate points within the desired range of the surface
            surf_array = np.concatenate((np.reshape(zz, (zz.size, 1)), np.reshape(yy, (yy.size, 1)), np.reshape(xx, (xx.size, 1))), axis=1)
            dist_array = distance_matrix(df[["Z", "Y", "X"]], surf_array)
            min_distance = np.min(dist_array, axis=1)
            # use cross product ot determine whether point is "above" or "below" surface
            min_ids = np.argmin(dist_array, axis=1)
            abv_list = []
            for i in range(len(min_ids)):
                zyx = df[["Z", "Y", "X"]].iloc[i]
                surf_zyx1 = surf_array[min_ids[i]]

                is_above = zyx[0] >= surf_zyx1[0]
                if is_above:
                    abv_list.append(1)
                else:
                    abv_list.append(-1)

            # highlight points
            min_distance = np.multiply(min_distance, abv_list)
            select_values = np.fromstring(select_range[1:-1], sep=',')
            pec_fin_nuclei = np.where((min_distance <= select_values[1]) & (min_distance >= select_values[0]))[0]

            if pec_fin_nuclei.any():
                f.add_trace(
                    go.Scatter3d(
                        mode='markers',
                        x=[df["X"].iloc[p] for p in pec_fin_nuclei],
                        y=[df["Y"].iloc[p] for p in pec_fin_nuclei],
                        z=[df["Z"].iloc[p] for p in pec_fin_nuclei],
                        marker=dict(
                            color='salmon',
                            size=5),
                        showlegend=False
                    )
                )
        return f, pec_fin_nuclei

    @app.callback(
        Output('save-button-hidden', 'children'),
        [Input('save-button', 'n_clicks'),
            Input('pfin_nuclei', 'children'),
            Input('not_fin_points', 'children'),
            Input('select-slider', 'value')])

    def clicks(n_clicks, pec_fin_nuclei, not_fin_points,slider_values):
        changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
        if 'save-button' in changed_id:
            lbl_vec = []
            for rg in range(df.shape[0]):
                if rg in pec_fin_nuclei:
                    lbl_vec.append(True)
                else:
                    lbl_vec.append(False)
            df.assign(PecFinFlag=np.asarray(lbl_vec))
            # save
            write_file = dataRoot + filename + '_nucleus_props.csv'
            df.to_csv(write_file)

            write_file2 = dataRoot + filename + '_curation_info/not_fin_points.pkl'
            with open(write_file2, 'wb') as wf:
                pickle.dump(not_fin_points, wf)

            write_file3 = dataRoot + filename + '_curation_info/value_slider.pkl'
            with open(write_file3, 'wb') as wf:
                pickle.dump(slider_values, wf)

    app.run_server(debug=True)

if __name__ == '__main__':

    # set parameters
    filename = "2022_12_15 HCR Hand2 Tbx5a Fgf10a_1.zarr"
    dataRoot = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/pecFin/HCR_Data/built_zarr_files/"
    level = 1

    # load image data
    segment_pec_fins(dataRoot)

    #nucleus_coordinates = pd.read_csv('/Users/nick/test.csv')
    #print(nucleus_coordinates)
