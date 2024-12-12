import numpy as np
import pandas as pd
url = 'https://drive.google.com/file/d/1cKvL6ZuqTAmMDBjbSERP-GviJQrJzDjY/view?usp=sharing'
path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
df = pd.read_csv(path)
catdf = df.select_dtypes(include=['object']).columns
finaldf = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], dtype=int)
numdf=finaldf.drop(['sex_female', 'sex_male','smoker_no', 'smoker_yes', 'region_northeast', 'region_northwest','region_southeast', 'region_southwest'], axis=1)
#END of testing data; TEMPORARY
print(catdf.head())