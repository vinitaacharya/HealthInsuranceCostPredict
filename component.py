import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px

app = dash.Dash(__name__)

#Jenna: process the entire dataset. Make a df named finaldf which we will take 
#Make a df called numdf that has all the numerical columns.





#Testing data: THIS IS TEMPORARY
import numpy as np
import pandas as pd
url = 'https://drive.google.com/file/d/1cKvL6ZuqTAmMDBjbSERP-GviJQrJzDjY/view?usp=sharing'
path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
df = pd.read_csv(path)
catdf = df.select_dtypes(include=['object']).columns
finaldf = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], dtype=int)
print(finaldf.head())
numdf=finaldf.drop(['sex_female', 'sex_male','smoker_no', 'smoker_yes', 'region_northeast', 'region_northwest','region_southeast', 'region_southwest'], axis=1)
catdf = df.select_dtypes(include=['object']).columns
#END of testing data; TEMPORARY






#Get Numerical Values
#numerical_columns = finaldf.select_dtypes(include=['float64', 'int64']).columns
#Define options for the dropdown
optionsNum = []
optionsCat = []
for i in numdf:
  optionsNum.append({'label': i, 'value' : i})
#options = ['Option 1', 'Option 2', 'Option 3']

for i in catdf:
  optionsCat.append(i)

app.layout = html.Div(
    className="container",
    children=[
       html.Div(
          className="selectTarget",
          children=[html.H3("Select Target Variable:"),
          dcc.Dropdown(
            id='dropdown',
            options=optionsNum,
            value=numdf.columns[0],
            style={'width': '150px'}
        )]),
        html.Div(id='output', style={'fontSize': '22px'}),  # Placeholder for output
        dcc.RadioItems(
           id='radio',
           options = optionsCat,
           value = df['sex'].unique()[0],
           inline=True
           
        ),
        dcc.Graph(id='corrBar')
    ]
)


#------------------------------------Callback block starts
@app.callback([Output('output', 'children'),Output('corrBar', 'figure')],[Input('dropdown', 'value')])
def update_output(selected_option):
    corr = numdf.corr()
    corr = corr.drop(selected_option, axis=0)
    targetcorr = corr[selected_option]
    
    corrdf = targetcorr.reset_index()
    corrdf.columns = ['Numerical Variables', 'Correlation Strength'] 
    fig = px.bar(corrdf, x='Numerical Variables', y='Correlation Strength', title=f'Correlation Strength of numerical variables with {selected_option}')

    return f'Average {selected_option} by', fig
#------------------------------------Callback block ends

if __name__ == '__main__':
    app.run(debug=True)
