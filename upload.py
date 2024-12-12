import dash
from dash import dcc, html, State
from dash.dependencies import Input, Output
import plotly.express as px
import numpy as np
import pandas as pd
import base64
import io
from sklearn.preprocessing import StandardScaler, OneHotEncoder

app = dash.Dash(__name__)

#-----UPLOAD COMPONENT------------------------------------------------------------
uploaded_data = None
numdf = None
catdf = None
processed_data = None

def generate_table(df):
    if df is None:
        return html.Div("No data to display.")
    
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in df.columns])] +
        # Body
        [html.Tr([html.Td(df.iloc[i][col]) for col in df.columns]) for i in range(min(len(df), 10))]
    )

app.layout = html.Div([
    html.Div([
        dcc.Upload(
            id='upload-data',
            children=html.Div([  # Button for uploading CSV
                html.Button("Upload CSV File", style={
                    "padding": "10px 20px",
                    "border": "2px solid #007BFF",
                    "background-color": "#007BFF",
                    "color": "white",
                    "font-size": "16px",
                    "cursor": "pointer",
                    "border-radius": "5px",
                    "text-align": "center"
                })
            ]),
            multiple=False,
            style={  # Upload box style
                "width": "100%",
                "border": "1px dashed #007BFF",
                "border-radius": "10px",
                "text-align": "center",
                "padding": "20px",
                "margin-top": "10px"
            }
        ),
        html.Div(id='upload-output', style={'marginTop': 10})
    ])
])

@app.callback(
    [Output('upload-output', 'children')],
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')]
)
def process_upload(contents, filename):
    global uploaded_data, numdf, catdf, processed_data
    if contents is None:
        return ["No file uploaded."]

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
    #this way was being difficult so I used a library preprocessing one instead
    #catdf = pd.get_dummies(catdf, columns=catdf.columns, drop_first=True)
    
    # Initialize OneHotEncoder
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    
    # Fit and transform categorical data using OneHotEncoder
    catdf_encoded = encoder.fit_transform(catdf)
    
    # Convert the result to a DataFrame with proper column names
    catdf_encoded = pd.DataFrame(catdf_encoded, columns=encoder.get_feature_names_out(catdf.columns))

    # Normalize/standardize numerical data
    scaler = StandardScaler()
    numdf_scaled = pd.DataFrame(scaler.fit_transform(numdf), columns=numdf.columns)

    # Combine processed data
    finaldf = pd.concat([numdf_scaled, catdf_encoded], axis=1)

    #THIS IS SO I CAN SEE THE DATA
    # Return all elements wrapped in a single Div container
    return [
        html.Div([
            html.H5(f"Uploaded Data ({filename})"),
            generate_table(uploaded_data),
            html.H5("Numerical Data (Imputed)"),
            generate_table(numdf_scaled),
            html.H5("Categorical Data (OG)"),
            generate_table(catdf),
            html.H5("Categorical Data (Imputed)"),
            generate_table(catdf_encoded),
            html.H5("Processed Data"),
            generate_table(finaldf)
        ])
    ]
#---------UPLOAD END--------------------------------------------------------

if __name__ == '__main__':
    app.run_server(debug=True)
