import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go

app = dash.Dash(__name__)

#Jenna: process the entire dataset. Make a df named finaldf which we will take 
#Make a df called numdf that has all the numerical columns.
#make a df called catdf that has all the categorical column names before encoding.





#Testing data: THIS IS TEMPORARY-----------------------------------------------
import numpy as np
import pandas as pd
url = 'https://drive.google.com/file/d/1cKvL6ZuqTAmMDBjbSERP-GviJQrJzDjY/view?usp=sharing'
path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
df = pd.read_csv(path)
catdf = df.select_dtypes(include=['object'])
finaldf = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], dtype=int)
numdf=finaldf.drop(['sex_female', 'sex_male','smoker_no', 'smoker_yes', 'region_northeast', 'region_northwest','region_southeast', 'region_southwest'], axis=1)
#END of testing data; TEMPORARY -------------------------------------




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
        html.Div(
           className="bargraphs",
           children = [
            html.Div(
              className="radioclass",
              children = [
                dcc.RadioItems(
                    id='radio',
                    options = optionsCat,
                    value = optionsCat[0],
                    inline=True),
                dcc.Graph(id='avgBar')]),
            html.Div(
               className="corrclass",
               children = [dcc.Graph(id='corrBar')]
            )
           ]
        ),
    ]
)


#------------------------------------Callback block starts
@app.callback([Output('corrBar', 'figure'), Output('avgBar', 'figure')],[Input('dropdown', 'value'), Input('radio', 'value')])
def update_output(selected_option, select_radio):
    corr = numdf.corr()
    corr = corr.drop(selected_option, axis=0)
    targetcorr = corr[selected_option]
    
    corrdf = targetcorr.reset_index()
    corrdf.columns = ['Numerical Variables', 'Correlation Strength'] 
    figCorr = px.bar(corrdf, x='Numerical Variables', y='Correlation Strength', text_auto=True)
    figCorr.update_layout(title_text=f'Correlation Strength of numerical variables with {selected_option}', title_x=0.5)

    avgdf = df.groupby(select_radio)[selected_option].mean()
    
    figAvg = px.bar(avgdf, y =selected_option, text_auto=True)
    figAvg.update_layout(title_text=f'Average {selected_option} by {select_radio}', title_x=0.5)
       

    return figCorr, figAvg
#------------------------------------Callback block ends

if __name__ == '__main__':
    app.run(debug=True)
