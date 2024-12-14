import dash
from dash import dcc, html, State
from dash.dependencies import Input, Output
import plotly.express as px
import numpy as np
import pandas as pd
import base64
import io
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from dash.exceptions import PreventUpdate

app = dash.Dash(__name__)

uploaded_data = None
numdf = None
catdf = None
finaldf = None
trained_model = None

def process_upload(contents):
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
    return uploaded_data, numdf, catdf, finaldf

app.layout = html.Div([
    # Upload Component
    html.Div([
        dcc.Upload(
            id='upload-data',
            children=html.Div([
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
        html.Div(id='upload-output', style={'marginTop': 10, "justify-content": "center"})
    ]),

    # Target Variable Selection and Bar Charts
    html.Div(
        className="container",
        children= [
            html.Div(
                className="selectTarget",
                children=[
                    html.H3("Select Target Variable:"),
                    dcc.Dropdown(id='dropdown', style={'width': '150px'})
                ]
            ),
            html.Div(
                className="bargraphs",
                children=[
                    html.Div(
                        className="radioclass",
                        children=[
                            dcc.RadioItems(id='radio', inline=True),
                            dcc.Graph(id='avgBar'),
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
    ),

    # Train Component
    html.Div(
        className="trainsec",
        children = [
            html.H3("Train Model"),
            html.Div(id='feature-checkboxes', style={'marginTop': '10px'}),
            html.Button("Train Model", id='train-button', className='btn', n_clicks=0),
            html.Div(id='train-output', style={'marginTop': '10px'})
    ]),

    # Predict Component
    html.Div(
        className="predictsec",
        children = [
            html.H3("Predict Target Variable"),
            dcc.Input(id='predict-input', placeholder='Enter feature values separated by commas', type='text', style={'width': '40%', 'margin' : '5px', 'border-color' : 'white'}),
            html.Button("Predict", id='predict-button', className='btn', n_clicks=0),
            html.Div(id='predict-output', style={'marginTop': '10px'})
    ])
])

@app.callback(
    [Output('upload-output', 'children'),
     Output('dropdown', 'options'),
     Output('dropdown', 'value'),
     Output('radio', 'options'),
     Output('radio', 'value')],
    [Input('upload-data', 'contents')]
)
def update_upload(contents):
    global uploaded_data, numdf, catdf, finaldf
    if contents is None:
        return ["No file uploaded"], [], None, [], None

    uploaded_data, numdf, catdf, finaldf = process_upload(contents)
    optionsNum = [{'label': col, 'value': col} for col in numdf.columns]
    optionsCat = [{'label': col, 'value': col} for col in catdf.columns]

    return ["File uploaded successfully"], optionsNum, optionsNum[0]['value'], optionsCat, optionsCat[0]['value']

@app.callback(
    [Output('corrBar', 'figure'), Output('avgBar', 'figure')],
    [Input('dropdown', 'value'), Input('radio', 'value')]
)
def update_charts(selected_option, select_radio):
    if finaldf is None:
        raise PreventUpdate

    corr = numdf.corr()[selected_option].drop(selected_option, errors='ignore')
    figCorr = px.bar(corr, text_auto=True)
    avgdf = uploaded_data.groupby(select_radio)[selected_option].mean().reset_index()
    figAvg = px.bar(avgdf, x=select_radio, y=selected_option, text_auto=True)
    return figCorr, figAvg

@app.callback(
    [Output('feature-checkboxes', 'children')],
    [Input('upload-data', 'contents')]
)
def update_feature_checkboxes(contents):
    if contents is None or finaldf is None:
        raise PreventUpdate

    checkboxes = [dcc.Checklist(
        options=[{'label': col, 'value': col} for col in finaldf.columns],
        id='feature-select',
        inline=True
    )]
    return [checkboxes]

@app.callback(
    [Output('train-output', 'children')],
    [Input('train-button', 'n_clicks')],
    [State('feature-select', 'value')]
)
def train_model(n_clicks, selected_features):
    global trained_model
    if n_clicks == 0 or finaldf is None or not selected_features:
        raise PreventUpdate

    X = finaldf[selected_features]
    y = uploaded_data[uploaded_data.columns[0]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline = Pipeline([('imputer', SimpleImputer(strategy='mean')), ('regressor', LinearRegression())])
    pipeline.fit(X_train, y_train)
    trained_model = pipeline
    r2_score = pipeline.score(X_test, y_test)
    return [f'Model trained successfully with R² score: {r2_score:.2f}']

@app.callback(
    [Output('predict-output', 'children')],
    [Input('predict-button', 'n_clicks')],
    [State('predict-input', 'value')]
)
def predict_value(n_clicks, input_values):
    if n_clicks == 0 or not input_values:
        raise PreventUpdate

    try:
        input_data = [float(x.strip()) for x in input_values.split(',')]
        prediction = trained_model.predict([input_data])
        return [f'Predicted Value: {prediction[0]:.2f}']
    except ValueError:
        return ['Invalid input. Please enter numeric values separated by commas.']

if __name__ == '__main__':
    app.run_server(debug=True)
