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

#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


# set parameters
filename = "2022_12_15 HCR Hand2 Tbx5a Fgf10a_1.zarr"
dataPath = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/pecFin/HCR_Data/built_zarr_files_small/" + filename
labelPath = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/pecFin/HCR_Data/built_zarr_files_small/" + filename + "labels"
level = 1

def load_image_data(dataPath, labelPath, fileName, level):

    #############
    # Main image
    #############

    # read the image data
    store = parse_url(dataPath, mode="r").store
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
    df = pd.DataFrame(centroid_array, columns=['x_val','z_val','y_val'])

    return df

########################
# App

app = dash.Dash(__name__)#, external_stylesheets=external_stylesheets)
#app.css.append_css({
#    "external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"
#})


def create_figure(skip_points=[]):
    dfs = df.drop(skip_points)
    return px.scatter_3d(dfs, x = 'x_val', y = 'y_val', z = 'z_val', opacity=0.5)
f= create_figure()

app.layout = html.Div([#html.Button('Delete', id='delete'),
                    html.Button('Clear Selection', id='clear'),
                    html.Button('Save', id='save-button'),
                    html.Button('Calculate', id='calc-button'),
                    html.P(id='save-button-hidden', style={'display': 'none'}),
                    dcc.Graph(id = '3d_scat', figure=f),
                    html.Div('selected:'),
                    html.Div(id='selected_points'), #, style={'display': 'none'})),
                    html.Div('deleted:'),
                    html.Div(id='deleted_points') #, style={'display': 'none'}))
])

# @app.callback(Output('deleted_points', 'children'),
#             [Input('delete', 'n_clicks')],
#             [State('selected_points', 'children'),
#             State('deleted_points', 'children')])
# def delete_points(n_clicks, selected_points, delete_points):
#     #print('n_clicks:',n_clicks)
#     if selected_points:
#         selected_points = json.loads(selected_points)
#     else:
#         selected_points = []
#
#     if delete_points:
#         deleted_points = json.loads(delete_points)
#     else:
#         deleted_points = []
#     ns = [p['pointNumber'] for p in selected_points]
#     new_indices = [df.index[n] for n in ns if df.index[n] not in deleted_points]
#     #print('new',new_indices)
#     deleted_points.extend(new_indices)
#     return json.dumps(deleted_points)


@app.callback(Output('selected_points', 'children'),
                [Input('3d_scat', 'clickData'),
                Input('deleted_points', 'children'),
                Input('clear', 'n_clicks')],
                [State('selected_points', 'children')])

def select_point(clickData, deleted_points, clear_clicked, selected_points):
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
    if 'deleted_points.children' in ids or 'clear.n_clicks' in ids:
        results = []
    results = json.dumps(results)
    #coordinates = [[p['x']] for p in results]

    return results

@app.callback(Output('3d_scat', 'figure'),
            [Input('selected_points', 'children'),
             Input('deleted_points',  'children')],
            [State('deleted_points', 'children')])

def chart_3d( selected_points, deleted_points_input, deleted_points_state):
    global f
    output_points = selected_points
    deleted_points = json.loads(deleted_points_state) if deleted_points_state else []
    f = create_figure(deleted_points)

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

    return f

@app.callback(
    Output('3d-surf', 'children'),
    [Input('calc-button', 'n_clicks'),
    Input('selected_points', 'children')])
def clicks(n_clicks, selected_points):
    if n_clicks != None:
        if n_clicks > 0:
            point_dict = json.loads(selected_points)
            coordinates = np.reshape([[p['z'], p['y'], p['x']] for p in point_dict],(len(point_dict), 3))
            # fit 3D surface
            A = np.c_[np.ones(data.shape[0]), coordinates[:, :2], np.prod(data[:, :2], axis=1), data[:, :2] ** 2, data[:, :2] ** 3]
            C, _, _, _ = scipy.linalg.lstsq(A, data[:, 2])


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

if __name__ == '__main__':


    # load image data
    nucleus_centroids = load_image_data(dataPath, labelPath, fileName, level)

    app.run_server(debug=True)

    nucleus_coordinates = pd.read_csv('/Users/nick/test.csv')
    print(nucleus_coordinates)
