import pandas as pd
import numpy as np

data = pd.read_csv('diabetes.csv')
data.head()

data.info()

data = data[1:768]

data = data.drop(['Unnamed: 0'],axis=1)

data.isnull().sum()

x = data.iloc[:,:-1]
 y = data.iloc[:,-1]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=25,random_state=0)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)

from sklearn.metrics import accuracy_score
acc_rand_forest = round(accuracy_score(y_pred,y_test),2)*100
print("Accuracy of random forest classifier is :",acc_rand_forest)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,r2_score,classification_report
logreg = LogisticRegression(solver = 'lbfgs',max_iter = 1000)
logreg.fit(x_train,y_train)
y_pred2 = logreg.predict(x_test)
acc_logreg = round(accuracy_score(y_pred2,y_test),2)*100
print("Accuracy of logistic regression  is :",acc_logreg)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(x_train,y_train)
y_pred3 = knn.predict(x_test)
acc_knn =  round(accuracy_score(y_pred3,y_test),2)*100
print("Accuracy of KNeighbors Classifier  is :",acc_knn)

