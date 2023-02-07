import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table_experiments as dt
import json
import pandas as pd
import plotly

app = dash.Dash()

app.scripts.config.serve_locally = True

DF_SIMPLE = pd.DataFrame({
    'x': ['A', 'B', 'C', 'D', 'E', 'F'],
    'y': [4, 3, 1, 2, 3, 6],
    'z': ['a', 'b', 'c', 'a', 'b', 'c']
})

app.layout = html.Div([
    html.H4('Editable DataTable'),
    dt.DataTable(
        rows=DF_SIMPLE.to_dict('records'),

        # optional - sets the order of columns
        columns=sorted(DF_SIMPLE.columns),

        filterable=True,
        sortable=True,
        editable=True,

        id='editable-table'
    ),
    html.Div([
        html.Button('Save', id='save-button'),
        html.P(id='editable-table-hidden', style={'display': 'none'}),
        html.P(id='save-button-hidden', style={'display': 'none'}),

    ], className='row')
], className='container')


@app.callback(
    Output('editable-table-hidden', 'children'),
    [Input('editable-table', 'rows')])
def update_selected_row_indices(rows):
    global DF_SIMPLE
    DF_SIMPLE = pd.DataFrame(rows)
    return json.dumps(rows, indent=2)


@app.callback(
    Output('save-button-hidden', 'children'),
    [Input('save-button', 'n_clicks')])
def clicks(n_clicks):
    if n_clicks > 0:
        DF_SIMPLE.to_csv('df.csv', index=False, encoding='utf-8')


if __name__ == '__main__':
    app.run_server(debug=True)