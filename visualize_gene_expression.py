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
import os


def segment_pec_fins(dataRoot):

    global fileList, imNameList
    # get list of filepaths
    fileList = sorted(glob.glob(dataRoot + '*_nucleus_props.csv'))
    imNameList = []
    for fn in range(len(fileList)):
        labelName = fileList[fn].replace(dataRoot, '', 1)
        labelName = labelName.replace('_nucleus_props.csv', '')
        imNameList.append(labelName)

    # compile dictionary of gene names that correspond to each dataset
    gene_name_dict = {}
    for f in range(len(fileList)):
        filename_temp = imNameList[f]
        propPath = dataRoot + filename_temp + '_nucleus_props.csv'

        if os.path.isfile(propPath):
            df_temp = pd.read_csv(propPath, index_col=0)
        else:
            raise Exception(
                f"Selected dataset( {filename_temp} ) dataset has no nucleus data. Have you run extract_nucleus_stats?")

        # get list of genes that we can look at
        colnames_temp = df_temp.columns
        list_raw = [item for item in colnames_temp if "_cell_mean_nn" in item]
        gene_names_temp = [item.replace("_cell_mean_nn", "") for item in list_raw]

        gene_name_dict[filename_temp] = gene_names_temp


    def load_nucleus_dataset(filename):
        # global fin_points_prev, not_fin_points_prev, class_predictions_curr, df, curationPath, propPath

        propPath = dataRoot + filename + '_nucleus_props.csv'

        if os.path.isfile(propPath):
            df = pd.read_csv(propPath, index_col=0)
        else:
            raise Exception(
                f"Selected dataset( {filename} ) dataset has no nucleus data. Have you run extract_nucleus_stats?")

        fin_nuclei = np.where(df["pec_fin_flag"] == 1)
        df = df.iloc[fin_nuclei]

        return df

    global df, colnames, gene_names

    df = load_nucleus_dataset(imNameList[0])

    # get list of genes that we can look at
    colnames = df.columns
    list_raw = [item for item in colnames if "_cell_mean_nn" in item]
    gene_names = [item.replace("_cell_mean_nn", "") for item in list_raw]

    ########################
    # App
    app = dash.Dash(__name__)  # , external_stylesheets=external_stylesheets)
    # global init_toggle
    # init_toggle = True

    def create_figure(df, gene_name=None):
        colormaps = ["ice", "inferno", "viridis"]

        if gene_name == None:
            plot_gene = gene_names[0] + "_mean"
            cmap = colormaps[0]
        else:
            plot_gene = gene_name + "_mean"
            g_index = gene_names.index(gene_name)
            cmap = colormaps[g_index]
        fig = px.scatter_3d(df, x="X", y="Y", z="Z", opacity=0.4, color=plot_gene, color_continuous_scale=cmap)
        fig.update_traces(marker=dict(size=6))
        #fig.update_layout(coloraxis_showscale=False)

        return fig

    f = create_figure(df)

    app.layout = html.Div([
                        dcc.Graph(id='3d_scat', figure=f),
                        html.Div(id='df_list', hidden=True),
                        html.Div([
                            dcc.Dropdown(imNameList, imNameList[0], id='dataset-dropdown'),
                        ],
                        style={'width': '30%', 'display': 'inline-block'}),
                        html.Div(id='dd-output-container', hidden=True),
                        html.Div([
                            dcc.Dropdown(id='gene-dropdown'),
                        ],
                            style={'width': '20%', 'display': 'inline-block'}),
                            html.Div(id='gene-output-container', hidden=True)
                        ]
                        )

    # update list of genes
    @app.callback(
        [dash.dependencies.Output('gene-dropdown', 'options'),
         dash.dependencies.Output('gene-dropdown', 'value')],
        [dash.dependencies.Input('dataset-dropdown', 'value')]
    )
    def update_date_dropdown(name):
        new_dict = [{'label': i, 'value': i} for i in gene_name_dict[name]]
        return new_dict, new_dict[0]['value']

    # @app.callback(
    #     Output('dd-output-container', 'children'),
    #     Input('dataset-dropdown', 'value')
    # )
    # def load_wrapper(value):
    #     return value
    #
    # @app.callback(
    #     Output('gene-output-container', 'children'),
    #     Input('gene-dropdown', 'value')
    # )
    # def load_wrapper(value):
    #     return value

    @app.callback(Output('3d_scat', 'figure'),
                  [Input('gene-dropdown', 'value'),
                   Input('dataset-dropdown', 'value')])

    def chart_3d(gene_name, fileName):

        global f, df, gene_names

        # check to see which values have changed
        changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

        df = load_nucleus_dataset(fileName)

        # get list of genes that we can look at
        colnames = df.columns
        list_raw = [item for item in colnames if "_cell_mean_nn" in item]
        gene_names = [item.replace("_cell_mean_nn", "") for item in list_raw]

        f = create_figure(df, gene_name)

        f.update_layout(uirevision="Don't change")
        return f

    app.run_server(debug=True)

if __name__ == '__main__':

    # set parameters
    dataRoot = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/pecFin/HCR_Data/built_zarr_files/"

    # load image data
    segment_pec_fins(dataRoot)