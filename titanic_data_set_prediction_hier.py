import pandas as pd
import numpy as np
from sklearn.cluster import MeanShift
from sklearn import preprocessing
df=pd.read_excel(r'C:\Users\naman\Downloads\titanic.xls')
df.fillna(0,inplace=True)
#print(df.head())
original_df=pd.DataFrame.copy(df)
df.drop(['name','home.dest','body'],1,inplace=True)
def handle_non_numerical_data(df):
    
    # handling non-numerical data: must convert.
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        #print(column,df[column].dtype)
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            
            column_contents = df[column].values.tolist()
            #finding just the uniques
            unique_elements = set(column_contents)
            # great, found them. 
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    # creating dict that contains new
                    # id per unique string
                    text_digit_vals[unique] = x
                    x+=1
            # now we map the new "id" vlaue
            # to replace the string. 
            df[column] = list(map(convert_to_int,df[column]))

    return df
df=handle_non_numerical_data(df)
#print(df.head())
x=np.array(df.drop(['survived'],1).astype(float))
x=preprocessing.scale(x)
y=np.array(df['survived'])
clf=MeanShift()
clf.fit(x)
labels=clf.labels_
number_of_clusters=clf.cluster_centers_
n_clusters=len(np.unique(labels))
original_df['survival_column']=np.nan
for i in range(len(x)):
	original_df['survival_column'].iloc[i]=labels[i]
peeps_survive={}
for i in range(n_clusters):
	temp_df=original_df[original_df['survival_column']==float(i)]
	survival_peeps=temp_df[temp_df['survived']==1]
	peep_survive=len(survival_peeps)/len(temp_df)
	peeps_survive[i]=peep_survive
print(peeps_survive)
print(original_df[original_df['survival_column']==1].describe())