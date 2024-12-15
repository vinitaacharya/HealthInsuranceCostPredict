import dash
from dash import dcc, html, State
from dash.dependencies import Input, Output
import plotly.express as px
import numpy as np
import pandas as pd
import base64
import io
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from dash.exceptions import PreventUpdate

app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Global Variables
uploaded_data = None
numdf = None
catdf = None
finaldf = None
trained_model = None

# Function to process uploaded file
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

# App Layout
app.layout = html.Div([
    # Upload Component
    html.Div([
        dcc.Upload(
            id='upload-data',
            children=html.Button("Upload CSV File Here", style={
                "padding": "10px 20px", "font-size": "20px", "cursor": "pointer"
            }),
            multiple=False
        ),
        html.Div(id='upload-output', style={'marginTop': 10})
    ]),

    # Target Variable Selection and Bar Charts
    html.Div([
        html.H3("Select Target Variable:"),
        dcc.Dropdown(id='dropdown', style={'width': '150px'}),
        dcc.RadioItems(id='radio', inline=True),
        dcc.Graph(id='avgBar'),
        dcc.Graph(id='corrBar')
    ]),

    # Train Component
    html.Div([
        html.H3("Train Model"),
        html.Div(id='feature-checkboxes', style={'marginTop': '10px'}),
        html.Button("Train Model", id='train-button', n_clicks=0),
        html.Div(id='train-output', style={'marginTop': '10px'})
    ]),

    # Predict Component
    html.Div([
        html.H3("Predict Target Variable"),
        dcc.Input(id='predict-input', placeholder='Enter feature values separated by commas', type='text', style={'width': '100%'}),
        html.Button("Predict", id='predict-button', n_clicks=0),
        html.Div(id='predict-output', style={'marginTop': '10px'})
    ])
])

# Upload callback
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

# Chart update callback
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

# Feature selection checklist callback
@app.callback(
    [Output('feature-checkboxes', 'children')],
    [Input('upload-data', 'contents')]
)
def update_feature_checkboxes(contents):
    if contents is None or finaldf is None:
        raise PreventUpdate

    checkboxes = dcc.Checklist(
        options=[{'label': col, 'value': col} for col in finaldf.columns],
        id='feature-select',
        inline=True
    )
    return [checkboxes]

# Train model callback
@app.callback(
    [Output('train-output', 'children')],
    [Input('train-button', 'n_clicks')],
    [State('feature-checkboxes', 'children'),
     State('feature-select', 'value')]
)
def train_model(n_clicks, feature_checkboxes, selected_features):
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

# Predict callback
@app.callback(
    [Output('predict-output', 'children')],
    [Input('predict-button', 'n_clicks')],
    [State('predict-input', 'value')]
)
def predict_value(n_clicks, input_values):
    global trained_model
    if n_clicks == 0 or not input_values:
        raise PreventUpdate

    if trained_model is None:
        return ["Error: Train the model before making predictions."]

    try:
        input_data = [float(x.strip()) for x in input_values.split(',')]
        prediction = trained_model.predict([input_data])
        return [f'Predicted Value: {prediction[0]:.2f}']
    except ValueError:
        return ['Invalid input. Please enter numeric values separated by commas.']
    except Exception as e:
        return [f"Error: {str(e)}"]

if __name__ == '__main__':
    app.run_server(debug=True)

