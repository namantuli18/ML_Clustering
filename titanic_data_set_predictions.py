import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate,train_test_split
from sklearn import preprocessing,svm,neighbors
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
df=pd.read_excel(r"C:\Users\naman\Downloads\titanic.xls")
##print(df.head())
def handle_non_numeric_data(df):
	columns=df.columns.values
	print(columns)
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
#print(df.tail())
df.drop(['ticket','name'],1,inplace=True)
x=np.array(df.drop(['survived'],1).astype(float))
x=preprocessing.scale(x)
y=np.array(df['survived'])
clf=svm.SVC()
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
clf.fit(x_train,y_train)
accuracy=clf.score(x_test,y_test)
print(accuracy)
print(df[df['survived']==1])
correct=0
predict_me=np.array([1,0,60,0,1,250,34,1,1,14,12])
predict_me=predict_me.reshape(-1,len(predict_me))
prediction=clf.predict(predict_me)
print(prediction)
'''for i in range(len(x)):
	predict_me=np.array(x[i].astype(float))
	predict_me=predict_me.reshape(-1,len(predict_me))
	prediction=clf.predict(predict_me)
	if prediction[0]==y[i]:
		correct+=1
print(correct/len(x))'''
