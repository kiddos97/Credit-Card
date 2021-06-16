# python ccode will detect credit card fraud
# few machine learnig and Data analysis packages are being used

import numpy as np # data processing
import pandas as pd # working with arrays
import matplotlib.pyplot as plt #data visualization
from termcolor import colored as cl # text customization
import itertools # advanced tools


from sklearn.preprocessing import StandardScaler # data normalization
from sklearn.model_selection import train_test_split # data split
from sklearn.tree import DecisionTreeClassifier # Decision tree algorithm
from sklearn.neighbors import KNeighborsClassifier # KNN algorithm
from sklearn.linear_model import LogisticRegression # Logistic regression algorithm
from sklearn.svm import SVC # SVM algorithm
from sklearn.ensemble import RandomForestClassifier # Random forest tree algorithm


from sklearn.metrics import confusion_matrix # evaluation metric
from sklearn.metrics import accuracy_score # evaluation metric
from sklearn.metrics import f1_score # evaluation metric

#data contains features V1 to V28

#importing the data

df = pd.read_csv('D:\\creditcard\\creditcard.csv')
df.drop('Time', axis = 1, inplace = True)

print(df)

# here we are going to look at how many fraud cases and non fraud cases we have in the data
# also we will compute the perfecntage of fraud cases
cases = len(df)
nonfraud = len(df[df.Class == 0])
fraudcount = len(df[df.Class == 1])
fraud_percent = round(fraudcount/nonfraud *100,2)

print(cl('CASE COUNT', attrs = ['bold']))
print(cl('------------------------------------', attrs= ['bold']))
print(cl('Total number of cases are: {} '.format(cases), attrs = ['bold']))
print(cl('Number of Non-fraud cases are: {} '.format(nonfraud), attrs = ['bold']))
print(cl('Number of Fraud cases are: {} '.format(fraudcount), attrs = ['bold']))
print(cl('Percentage of Fraud cases are: {} '.format(fraud_percent), attrs = ['bold']))
print(cl('------------------------------------------ ', attrs = ['bold']))


#getting a statsical view of both fraud and non-faud transactions amount
nonfraud = df[df.Class == 0]
fraudcount = df[df.Class == 1]
print(cl('CASE AMOUNT STATISTICS',attrs = ['bold']))
print(cl('-------------------------------------------',attrs = ['bold']))
print(cl('NON-FRAUD CASE AMOUNT STATS', attrs = ['bold']))
print(nonfraud.Amount.describe())
print(cl('----------------------------------------',attrs = ['bold']))
print(cl('FRAUD CASE AMOUNT STATS', attrs = ['bold']))
print(fraudcount.Amount.describe())
print(cl('--------------------------------------------',attrs = ['bold']))

#reducing the wide range of values in the Amount variable
sc = StandardScaler()
amount = df['Amount'].values
df['Amount'] = sc.fit_transform(amount.reshape(-1,1))

print(cl(df['Amount'].head(10),attrs = ['bold']))
print(cl('-------------------------------------------------', attrs = ['bold']))
#splitting the data into two sets, training set and testing set
#Data split
print(cl('TEST AND TRAINING SAMPLES', attrs = ['bold']))
print(cl('----------------------------------------------',attrs = ['bold']))

X = df.drop('Class', axis = 1).values
Y = df['Class'].values
X_train,X_test, Y_train,Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 0)

print(cl('X_train samples : ',attrs = ['bold']),X_train[:1])
print(cl('X_test samples : ',attrs = ['bold']),X_test[:1])
print(cl('Y_train samples : ',attrs = ['bold']),Y_train[0:10])
print(cl('Y_test samples : ', attrs = ['bold']),Y_test[0:10])

#MODELING DATA
#Using various types of classification models to determine which is the
#most suitale one for our case

# 1. Decision Tree
#model will predict the value of the target variable based on the inputs of several other variables
tree_model = DecisionTreeClassifier(max_depth= 4, criterion= 'entropy')
tree_model.fit(X_train,Y_train)
tree_yhat = tree_model.predict(X_test)

# 2. K-Nearesr Neighbors
# model is looking at variables in close proximity that are/ or could be similar
# beleiving that similar things are in close proxmity
n = 5
knn = KNeighborsClassifier(n_neighbors = n)
knn.fit(X_train,Y_train)
knn_yhat = knn.predict(X_test)

# 3. Logistic Regression
# this algorithm is based on the concept of probability
lr = LogisticRegression()
lr.fit(X_train,Y_train)
lr_yhat = lr.predict(X_test)


#4. SVM

svm = SVC()
svm.fit(X_train,Y_train)
svm_yhat = svm.predict(X_test)

#5. Random Forest Tree

rf = RandomForestClassifier(max_depth = 4)
rf.fit(X_train,Y_train)
rf_yhat = rf.predict(X_test)


#Accruacy Score
#here will get the accuracy of each classification model to see which one is the best fit for cause
print(cl('ACCURACU SCORE',attrs = ['bold']))
print(cl('--------------------------------------', attrs = ['bold']))
print(cl(' Accuracuy score of the Decision Tree model {}'.format(accuracy_score(Y_test,tree_yhat)),attrs = ['bold']))
print(cl(' Accuracy score of the KNN model is {}'.format(accuracy_score(Y_test, knn_yhat)),attrs = ['bold']))
print(cl('Accuracy score of the Logicstic Regression model is {}'.format(accuracy_score(Y_test,lr_yhat)),attrs = ['bold']))
