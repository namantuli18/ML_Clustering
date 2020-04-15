import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate,train_test_split
from sklearn import preprocessing
from sklearn.cluster import KMeans
df = pd.read_csv(r"C:\Users\naman\Downloads\winequality-red.csv")
##print(df.head())
def handle_non_numeric_data(df):
	columns=df.columns.values
	for column in columns:
		text_digit_vals={}
		def convert_to_int(val):
			return text_digit_vals[val]
		if df[column].dtype!=np.int64 and df[column].dtype!=np.float64:
			column_contents=df[column].values.tolist()
			unique_elements=set(column_contents)

			x=0
			for unique in unique_elements:
				if unique not in text_digit_vals:
					text_digit_vals[unique]=x
					x+=1
			df[column]=list(map(convert_to_int,df[column]))
	return df

df=handle_non_numeric_data(df)
df.fillna(0,inplace=True)
print(df.iloc[1])

x=np.array(df.drop(['density'],1))
x=preprocessing.scale(x)
y=np.array(df['pH'])
clf=KMeans(n_clusters=2)
clf.fit(x)
correct=0
for i in range(len(x)):
	predict_me=np.array(x[i].astype(float))
	predict_me=predict_me.reshape(-1,len(predict_me))
	prediction=clf.predict(predict_me)
	if prediction[0]==y[i]:
		correct+=1
print(correct/len(x))
