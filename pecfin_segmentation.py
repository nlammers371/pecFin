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
from skimage.measure import label, regionprops, regionprops_table
import itertools
from scipy.spatial import distance_matrix
#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


def segment_pec_fins(dataPath, labelPath, level):
    fileName = "2022_12_15 HCR Hand2 Tbx5a Fgf10a_1.zarr"

    def load_image_data(dataPath, labelPath, level):

        #############
        # Main image
        #############

        # read the image data
        #store = parse_url(dataPath, mode="r").store
        reader = Reader(parse_url(dataPath))

        # nodes may include images, labels etc
        nodes = list(reader())

        # first node will be the image pixel data
        image_node = nodes[0]
        image_data = image_node.data

        #############
        # Labels
        #############

        # read the image data
        store_lb = parse_url(labelPath, mode="r").store
        reader_lb = Reader(parse_url(labelPath))

        # nodes may include images, labels etc
        nodes_lb = list(reader_lb())

        # first node will be the image pixel data
        label_node = nodes_lb[1]
        label_data = label_node.data

        # extract key image attributes
        #omero_attrs = image_node.root.zarr.root_attrs['omero']
        #channel_metadata = omero_attrs['channels']  # list of channels and relevant info
        multiscale_attrs = image_node.root.zarr.root_attrs['multiscales']
        #axis_names = multiscale_attrs[0]['axes']
        #dataset_info = multiscale_attrs[0]['datasets']  # list containing scale factors for each axis

        # extract useful info
        scale_vec = multiscale_attrs[0]["datasets"][level]["coordinateTransformations"][0]["scale"]

        # add layer of mask centroids
        label_array = np.asarray(label_data[level].compute())
        regions = regionprops(label_array)

        centroid_array = np.empty((len(regions), 3))
        for rgi, rg in enumerate(regions):
            centroid_array[rgi, :] = rg.centroid

        centroid_array = np.multiply(centroid_array, scale_vec)
        df = pd.DataFrame(centroid_array, columns=['z_val','y_val','x_val'])

        return df

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
    # get data frame
    df = load_image_data(dataPath, labelPath, level)
    # define mesh grid for curve fitting
    nx = 100
    ny = 100

    global xx, yy, xm
    xm = np.max(df["x_val"])
    xx, yy = np.meshgrid(np.linspace(0, xm, nx),
                         np.linspace(0, np.max(df["y_val"]), ny))

    ########################
    # App

    app = dash.Dash(__name__)#, external_stylesheets=external_stylesheets)
    #app.css.append_css({
    #    "external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"
    #})


    def create_figure(skip_points=[]):
        dfs = df.drop(skip_points)
        return px.scatter_3d(dfs, x='x_val', y='y_val', z='z_val', opacity=0.5)
    f= create_figure()

    app.layout = html.Div([#html.Button('Delete', id='delete'),
                        html.Button('Clear Selection', id='clear'),
                        html.Button('Save', id='save-button'),
                        html.Button('Calculate Fit', id='calc-button'),
                        html.P(id='save-button-hidden', style={'display': 'none'}),
                        dcc.Graph(id='3d_scat', figure=f),
                        html.Div('selected:'),
                        html.Div(id='selected_points'),
                        dcc.RangeSlider(-round(0.25*xm), round(0.25*xm), id='select-slider'),
                        html.Div(id='output-container-range-slider')
    ])


    @app.callback(Output('selected_points', 'children'),
                    [Input('3d_scat', 'clickData'),
                    Input('clear', 'n_clicks')],
                    [State('selected_points', 'children')])

    def select_point(clickData,  n_clicks, selected_points):
        ctx = dash.callback_context
        ids = [c['prop_id'] for c in ctx.triggered]

        if selected_points:
            results = json.loads(selected_points)
        else:
            results = []


        if '3d_scat.clickData' in ids:
            if clickData:
                for p in clickData['points']:
                    if p not in results:
                        results.append(p)

        if n_clicks!=None:
            if n_clicks > 0:
                results = []

        results = json.dumps(results)
        #coordinates = [[p['x']] for p in results]

        return results

    @app.callback(
        Output('output-container-range-slider', 'children'),
        Input('select-slider', 'value'))
    def update_output(value):
        value = json.dumps(value)
        return value

    @app.callback(Output('3d_scat', 'figure'),
                [Input('selected_points', 'children'),
                 Input('output-container-range-slider', 'children'),
                 Input('calc-button', 'n_clicks'),
                 Input('select-slider', 'value')])

    def chart_3d(selected_points, select_range, n_clicks, slider_range):
        global f

        # check to see which values have changed
        changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

        f = create_figure()

        selected_points = json.loads(selected_points) if selected_points else []
        if selected_points:
            f.add_trace(
                go.Scatter3d(
                    mode='markers',
                    x=[p['x'] for p in selected_points],
                    y=[p['y'] for p in selected_points],
                    z=[p['z'] for p in selected_points],
                    marker=dict(
                        color='red',
                        size=5,
                        line=dict(
                            color='red',
                            width=2
                        )
                    ),
                    showlegend=False
                )
            )

        # check if calc button has been clicked
        # clear_fit = False
        # if n_clicks_fit != None:
        #     if n_clicks_fit > 0:
        #         clear_fit = True
        if selected_points and (('select-slider' in changed_id) | ('calc-button' in changed_id)):

            #oint_dict = json.loads(selected_points)
            xyz = np.reshape([[p['x'], p['y'], p['z']] for p in selected_points], (len(selected_points), 3))
            # Fit a 3rd order, 2d polynomial
            m = polyfit2d(xyz[:, 0], xyz[:, 1], xyz[:, 2], order=1)
            # Evaluate it on a grid...
            zz = polyval2d(xx, yy, m, )
            # add to figure
            f.add_trace(go.Surface(z=zz, x=xx, y=yy, colorscale='algae', opacity=0.5))

            # now calculate points within the desired range of the surface
            surf_array = np.concatenate((np.reshape(zz, (zz.size, 1)), np.reshape(yy, (yy.size, 1)), np.reshape(xx, (xx.size, 1))), axis=1)
            dist_array = distance_matrix(df[['z_val', 'y_val', 'x_val']], surf_array)
            min_distance = np.min(dist_array, axis=1)
            # use cross product ot determine whether point is "above" or "below" surface
            min_ids = np.argmin(dist_array, axis=1)
            abv_list = []
            for i in range(len(min_ids)):
                zyx = df[['z_val', 'y_val', 'x_val']].iloc[i]
                surf_zyx1 = surf_array[min_ids[i]]

                is_above = zyx[0] >= surf_zyx1[0]
                if is_above:
                    abv_list.append(1)
                else:
                    abv_list.append(-1)

            # highlight points
            min_distance = np.multiply(min_distance, abv_list)
            select_values = np.fromstring(select_range[1:-1], sep=',')
            idx_candidates = np.where((min_distance <= select_values[1]) & (min_distance >= select_values[0]))[0]

            if idx_candidates.any():
                f.add_trace(
                    go.Scatter3d(
                        mode='markers',
                        x=[df['x_val'].iloc[p] for p in idx_candidates],
                        y=[df['y_val'].iloc[p] for p in idx_candidates],
                        z=[df['z_val'].iloc[p] for p in idx_candidates],
                        marker=dict(
                            color='red',
                            size=5,
                            line=dict(
                                color='red',
                                width=2
                            )
                        ),
                        showlegend=False
                    )
                )
        return f

    # @app.callback(
    #     Output('3d_scat', 'figure'),
    #     [Input('3d_scat', 'figure'),
    #     Input('calc-button', [p['z']'n_clicks'),
    #     Input('selected_points', 'children')])
    # def clicks(n_clicks, selected_points, f):
    #     if n_clicks != None:
    #         print(n_clicks)
    #         if n_clicks > 0:
    #             point_dict = json.loads(selected_points)
    #             coordinates = np.reshape([[p['z'], p['y'], p['x']] for p in point_dict],(len(point_dict), 3))
    #
    #             xyz = np.roll(coordinates, 2, axis=1)
    #
    #             # Fit a 3rd order, 2d polynomial
    #             m = polyfit2d(xyz[:, 0], xyz[:, 1], xyz[:, 2], order=2)
    #             # Evaluate it on a grid...
    #             nx, ny = 100, 100
    #             xx, yy = np.meshgrid(np.linspace(xyz[:, 0].min(), xyz[:, 0].max(), nx), np.linspace(xyz[:, 1].min(), xyz[:, 1].max(), ny))
    #             zz = polyval2d(xx, yy, m)
    #             print(zz)


    @app.callback(
        Output('save-button-hidden', 'children'),
        [Input('save-button', 'n_clicks'),
        Input('selected_points', 'children')])
    def clicks(n_clicks, selected_points):
        if n_clicks != None:
            if n_clicks > 0:
                point_dict = json.loads(selected_points)
                coordinates = pd.DataFrame(np.reshape([[p['z'], p['y'], p['x']] for p in point_dict],(len(point_dict), 3)), columns=["Z", "Y", "X"])
                coordinates.to_csv('/Users/nick/test.csv', index=False, encoding='utf-8')

    app.run_server(debug=True)

if __name__ == '__main__':

    # set parameters
    filename = "2022_12_15 HCR Hand2 Tbx5a Fgf10a_1.zarr"
    dataPath = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/pecFin/HCR_Data/built_zarr_files_small/" + filename
    labelPath = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/pecFin/HCR_Data/built_zarr_files_small/" + filename + "labels"
    level = 1

    # load image data
    nucleus_centroids = segment_pec_fins(dataPath, labelPath, level)



    nucleus_coordinates = pd.read_csv('/Users/nick/test.csv')
    print(nucleus_coordinates)
