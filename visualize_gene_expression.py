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
from scipy.interpolate import LinearNDInterpolator
import alphashape

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

        return df

    global df, colnames, gene_names

    df = load_nucleus_dataset(imNameList[0])

    # get list of genes that we can look at
    colnames = df.columns
    list_raw = [item for item in colnames if "_cell_mean_nn" in item]
    gene_names = [item.replace("_cell_mean_nn", "") for item in list_raw]

    plot_list = ["3D Scatter", "Volume Plot", "Multiplot"]
    ########################
    # App
    app = dash.Dash(__name__)  # , external_stylesheets=external_stylesheets)
    # global init_toggle
    # init_toggle = True

    def create_figure(df, gene_name=None, plot_type=None):
        colormaps = ["ice", "inferno", "viridis"]

        if gene_name == None:
            plot_gene = gene_names[0] + "_mean_nn"
            cmap = colormaps[0]
        else:
            plot_gene = gene_name + "_mean_nn"
            g_index = gene_names.index(gene_name)
            cmap = colormaps[g_index]

        xyz_array = np.asarray(df[["X", "Y", "Z"]])

        if (plot_type == None) | (plot_type == "3D Scatter"):
            high_flags = df[plot_gene] >= 0.3
            low_flags = df[plot_gene] < 0.3
            fig = px.scatter_3d(df.iloc[np.where(high_flags)], x="X", y="Y", z="Z", opacity=0.5, color=plot_gene,
                                color_continuous_scale=cmap, range_color=(0, 1))

            fig.update_traces(marker=dict(size=8
                                          ))

            fig.add_trace(go.Scatter3d(x=df["X"].iloc[np.where(low_flags)],
                                       y=df["Y"].iloc[np.where(low_flags)],
                                       z=df["Z"].iloc[np.where(low_flags)],
                                       mode='markers',
                                       marker=dict(
                                           size=5,
                                           color=df[plot_gene].iloc[np.where(low_flags)],  # set color to an array/list of desired values
                                           colorscale=cmap,  # choose a colorscale
                                           opacity=0.1,
                                           cmin=0,
                                           cmax=1)
                                       ))
            fig.update_layout(coloraxis_colorbar_title_text='Normalized Expression')
            fig.update_layout(showlegend=False)
            # fig = px.scatter_3d(df.iloc[high_flags], x="X", y="Y", z="Z", opacity=0.6, size=plot_gene, color=plot_gene, color_continuous_scale=cmap)
            # fig.update_traces(marker=dict(size=8))

            fig.add_trace(go.Mesh3d(x=xyz_array[:, 0], y=xyz_array[:, 1], z=xyz_array[:, 2],
                                    alphahull=9,
                                    opacity=0.1,
                                    color='gray'))

        elif plot_type == "Volume Plot":
            
            # generate points to interpolate
            xx = np.linspace(min(df["X"]), max(df["X"]), num=30)
            yy = np.linspace(min(df["Y"]), max(df["Y"]), num=30)
            zz = np.linspace(min(df["Z"]), max(df["Z"]), num=30)

            X, Y, Z = np.meshgrid(xx, yy, zz)  # 3D grid for interpolation

            # generate normalized arrays
            X_norm = X / np.max(X)
            Y_norm = Y / np.max(Y)
            Z_norm = Z / np.max(Z)

            # generate interpolator
            gene_values = np.asarray(df[plot_gene])
            interp = LinearNDInterpolator(xyz_array, gene_values)

            # get interpolated estimate of gene expression
            G = interp(X, Y, Z)

            # generate alpha shape
            xyz_array_norm = np.divide(xyz_array, np.asarray([np.max(X), np.max(Y), np.max(Z)]))
            alpha_fin = alphashape.alphashape(xyz_array_norm, 9)
            xyz_long = np.concatenate((np.reshape(X_norm, (X.size, 1)),
                                       np.reshape(Y_norm, (Y.size, 1)),
                                       np.reshape(Z_norm, (Z.size, 1))),
                                      axis=1)
            inside_flags = alpha_fin.contains(xyz_long)
            G_long = G.flatten()
            G_long[~inside_flags] = np.nan

            fig = go.Figure(
                data=go.Volume(
                x=X.flatten(),
                y=Y.flatten(),
                z=Z.flatten(),
                value=G_long,
                opacity=.1,
                isomin=0.2,
                isomax=0.8,  # needs to be small to see through all surfaces
                surface_count=15,  # needs to be a large number for good volume rendering
                colorscale=cmap
            ))

            fig.add_trace(go.Mesh3d(x=xyz_array[:, 0], y=xyz_array[:, 1], z=xyz_array[:, 2],
                                    alphahull=9,
                                    opacity=0.1,
                                    color='gray'))

            fig.update_layout(coloraxis_colorbar_title_text='Normalized Expression')

        elif plot_type == "Multiplot":
            # fig = px.scatter_3d(df, x="X", y="Y", z="Z", opacity=0.4)
            # raise Warning("Plot type " + plot_type + " is not currently supported")
            # generate points to interpolate
            xx = np.linspace(min(df["X"]), max(df["X"]), num=25)
            yy = np.linspace(min(df["Y"]), max(df["Y"]), num=25)
            zz = np.linspace(min(df["Z"]), max(df["Z"]), num=25)

            X, Y, Z = np.meshgrid(xx, yy, zz)  # 3D grid for interpolation

            # generate normalized arrays
            X_norm = X / np.max(X)
            Y_norm = Y / np.max(Y)
            Z_norm = Z / np.max(Z)

            # generate interpolator
            gene_values1 = np.asarray(df[gene_names[0] + "_mean_nn"])
            interp1 = LinearNDInterpolator(xyz_array, gene_values1)

            gene_values2 = np.asarray(df[gene_names[1] + "_mean_nn"])
            interp2 = LinearNDInterpolator(xyz_array, gene_values2)

            gene_values3 = np.asarray(df[gene_names[2] + "_mean_nn"])
            interp3 = LinearNDInterpolator(xyz_array, gene_values3)

            # get interpolated estimate of gene expression
            G1 = interp1(X, Y, Z)
            G2 = interp2(X, Y, Z)
            G3 = interp3(X, Y, Z)

            # generate alpha shape
            xyz_array_norm = np.divide(xyz_array, np.asarray([np.max(X), np.max(Y), np.max(Z)]))
            alpha_fin = alphashape.alphashape(xyz_array_norm, 9)
            xyz_long = np.concatenate((np.reshape(X_norm, (X.size, 1)),
                                       np.reshape(Y_norm, (Y.size, 1)),
                                       np.reshape(Z_norm, (Z.size, 1))),
                                      axis=1)

            xyz_long2 = np.concatenate((np.reshape(X, (X.size, 1)),
                                        np.reshape(Y, (Y.size, 1)),
                                        np.reshape(Z, (Z.size, 1))),
                                        axis=1)

            inside_flags = alpha_fin.contains(xyz_long)

            G1_long = G1.flatten()
            G1_long[~inside_flags] = np.nan
            G1_indices = np.where(G1_long >= 0.4)

            G2_long = G2.flatten()
            G2_long[~inside_flags] = np.nan
            G2_indices = np.where(G2_long >= 0.4)

            G3_long = G3.flatten()
            G3_long[~inside_flags] = np.nan
            G3_indices = np.where(G3_long >= 0.4)



            fig = go.Figure(data=go.Mesh3d(x=xyz_long2[G1_indices, 0][0], y=xyz_long2[G1_indices, 1][0], z=xyz_long2[G1_indices, 2][0],
                                    alphahull=9,
                                    opacity=0.2,
                                    color='blue',
                                    name=gene_names[0]))

            #print(xyz_long2[G1_indices, 0])
            fig.add_trace(go.Mesh3d(x=xyz_long2[G2_indices, 0][0], y=xyz_long2[G2_indices, 1][0], z=xyz_long2[G2_indices, 2][0],
                                    alphahull=15,
                                    opacity=0.2,
                                    color='red',
                                    name=gene_names[1]))

            fig.add_trace(
                go.Mesh3d(x=xyz_long2[G3_indices, 0][0], y=xyz_long2[G3_indices, 1][0], z=xyz_long2[G3_indices, 2][0],
                          alphahull=9,
                          opacity=0.2,
                          color='green',
                          name=gene_names[2]))

            fig.add_trace(go.Mesh3d(x=xyz_array[:, 0], y=xyz_array[:, 1], z=xyz_array[:, 2],
                                    alphahull=9,
                                    opacity=0.1,
                                    color='gray'))

            fig.update_layout(showlegend=True)
            fig.update_layout(legend_title_text='Gene Name')

        else:
            fig = px.scatter_3d(df, x="X", y="Y", z="Z", opacity=0.4)
            raise Warning("Plot type " + plot_type + " is not currently supported")

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
                            html.Div(id='gene-output-container', hidden=True),

                        # html.Div([
                        #     dcc.Dropdown(plot_list, id='plot-dropdown'),
                        #     html.Div(id='plot-output-container')
                        # ])
                        html.Div(id='plot_list', hidden=True),
                        html.Div([
                            dcc.Dropdown(plot_list, plot_list[0], id='plot-dropdown'),
                        ],
                        style={'width': '20%', 'display': 'inline-block'}),
                        html.Div(id='plot-output-container', hidden=True)
                        ])

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
                   Input('dataset-dropdown', 'value'),
                   Input('plot-dropdown', 'value')])

    def chart_3d(gene_name, fileName, plot_type):

        global f, df, gene_names

        # check to see which values have changed
        changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

        df = load_nucleus_dataset(fileName)

        # get list of genes that we can look at
        colnames = df.columns
        list_raw = [item for item in colnames if "_cell_mean_nn" in item]
        gene_names = [item.replace("_cell_mean_nn", "") for item in list_raw]

        f = create_figure(df, gene_name, plot_type)

        f.update_layout(uirevision="Don't change")
        return f

    app.run_server(debug=True)

if __name__ == '__main__':

    # set parameters
    dataRoot = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/pecFin/HCR_Data/built_zarr_files/"

    # load image data
    segment_pec_fins(dataRoot)