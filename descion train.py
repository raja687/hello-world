import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics, tree

df=pd.read_csv('diabetes.csv')
print(df.head())

x=df.iloc[:,:-1]
y=df.iloc[:,-1]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=2)


df=tree.DecisionTreeclassifier(criterion='entropy',max_depth=3,random_state=0)
df.fit(x_train,y_train)

y_pred=df.predict(x_test)
print(y_pred)
print('Accuracy',metrics.accuracy_score(y_test,y_pred))



