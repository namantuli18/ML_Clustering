import numpy as np
from sklearn import preprocessing, neighbors,svm
import pandas as pd
from sklearn.model_selection import cross_validate,train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv(r"C:\Users\naman\Downloads\breast-cancer-wisconsin (2).data")
df.replace('?',-99999, inplace=True)
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'], 1))
#preprocessing.scale(X)
y = np.array(df['class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(df.iloc[1])
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)

print(accuracy)