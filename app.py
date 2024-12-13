import dash
from dash import dcc, html, State
from dash.dependencies import Input, Output
import plotly.express as px
import numpy as np
import pandas as pd
import base64
import io
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder

app = dash.Dash(__name__)

# Jenna: Process the entire dataset. Make a df named finaldf which we will take. 
# Make a df called numdf that has all the numerical columns.
# Make a df called catdf that has all the categorical column names before encoding.

#-----UPLOAD COMPONENT------------------------------------------------------------
uploaded_data = None
numdf = None
catdf = None
finaldf = None

app.layout = html.Div([
    html.Div([
        dcc.Upload(
            id='upload-data',
            children=html.Div([  # Button for uploading CSV
                html.Button("Upload CSV File Here", style={
                    "padding": "10px 20px",
                    "color": "rgb(165, 113, 199)",
                    "font-size": "20px",
                    "letter-spacing": "1px",
                    "font-weight": "4px",
                    "cursor": "pointer",
                    "text-align": "center",
                    "border" : "0px",
                    "border-radius": "10px",
                    "font-family" : "'Lucida Sans', 'Lucida Sans Regular', 'Lucida Grande', 'Lucida Sans Unicode', Geneva, Verdana, sans-serif",
                    "background-color" : "#faf9f9"
                })
            ]),
            multiple=False,
            style={  # Upload box style
                "width": "100%",
                "border": "2px dashed rgb(165, 113, 199)",
                "border-radius" : "5px",
                "text-align": "center",
                "padding": "20px",
                "margin-top": "10px",
                "background-color" : "#faf9f9"
            }
        ),
        html.Div(id='upload-output', style={'marginTop': 10})
    ]),
    # Target variable and bar charts
    html.Div(
        className="container",
        children=[
            # Select Target Component (10 points): Contains a label and dropdown for selecting the target variable
            html.Div(
                className="selectTarget",
                children=[
                    html.H3("Select Target Variable:"),
                    dcc.Dropdown(id='dropdown', style={'width': '150px', 'color': 'black'}),
                ]
            ),
            # Bar charts components (30 points): Analyze dataset using two bar charts
            html.Div(
                className="bargraphs",
                children=[
                    html.Div(
                        className="radioclass",
                        children=[
                            dcc.RadioItems(id='radio', inline=True),
                            dcc.Graph(id='avgBar')
                        ]
                    ),
                    html.Div(
                        className="corrclass",
                        children=[
                            dcc.Graph(id='corrBar')
                        ]
                    )
                ]
            )
        ]
    )
])

@app.callback(
    [Output('upload-output', 'children'),
     Output('dropdown', 'options'),
     Output('dropdown', 'value'),
     Output('radio', 'options'),
     Output('radio', 'value')],
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')]
)
def process_upload(contents, filename):
    global uploaded_data, numdf, catdf, finaldf
    if contents is None:
        return [""], [], None, [], None

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string).decode('utf-8')
    uploaded_data = pd.read_csv(io.StringIO(decoded))

    # Separate numerical and categorical data
    numdf = uploaded_data.select_dtypes(include=['number'])
    catdf = uploaded_data.select_dtypes(exclude=['number'])

    # Handle missing values
    numdf = numdf.fillna(numdf.median())
    catdf = catdf.fillna(catdf.mode().iloc[0])

    # Encode categorical variables
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    catdf_encoded = pd.DataFrame(encoder.fit_transform(catdf))
    catdf_encoded.columns = encoder.get_feature_names_out(catdf.columns)

    # Normalize/standardize numerical data
    scaler = StandardScaler()
    numdf_scaled = pd.DataFrame(scaler.fit_transform(numdf), columns=numdf.columns)

    # Combine processed data
    finaldf = pd.concat([numdf_scaled, catdf_encoded], axis=1)

    # Generate dropdown and radio options
    optionsNum = [{'label': col, 'value': col} for col in numdf.columns]
    optionsCat = [{'label': col, 'value': col} for col in catdf.columns]

    return [""], optionsNum, optionsNum[0]['value'], optionsCat, optionsCat[0]['value']

#------------------------------------Callback block starts
@app.callback(
    [Output('corrBar', 'figure'), Output('avgBar', 'figure')],
    [Input('dropdown', 'value'), Input('radio', 'value')]
)
def update_output(selected_option, select_radio):
    global finaldf
    if finaldf is None:
        return {}, {}

    # Correlation graph
    corr = numdf.corr()
    targetcorr = corr[selected_option].drop(selected_option, errors='ignore')
    corrdf = targetcorr.reset_index()
    corrdf.columns = ['Numerical Variables', 'Correlation Strength']
    figCorr = px.bar(corrdf, x='Numerical Variables', y='Correlation Strength', text_auto=True)
    figCorr.update_layout(title_text=f'Correlation Strength of numerical variables with {selected_option}', title_x=0.5)

    # Average graph
    avgdf = uploaded_data.groupby(select_radio)[selected_option].mean().reset_index()
    figAvg = px.bar(avgdf, x=select_radio, y=selected_option, text_auto=True)
    figAvg.update_layout(title_text=f'Average {selected_option} by {select_radio}', title_x=0.5)

    return figCorr, figAvg
#------------------------------------Callback block ends

if __name__ == '__main__':
    app.run_server(debug=True)


    