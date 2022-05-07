#Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

#Loading data into dataframe(df)
df=pd.read_csv('Lung_cancer_data.csv')

print(df.head(10))#Print all data of top 10 rows
print(df.shape)#Print the row and clumn count of the data
print(df.isna().sum())#Print all columns with empty data along with sum of empty data

df=df.dropna(axis=1)#Drop the column with empty data
print(df.iloc[:,0].value_counts())#Visualize the data of diagnosis column with y label counts
print(df.dtypes)#Data type of data in each column

#Splitting data for dependence
X=df.iloc[:,1:].values#Features of cancerous and non cancerous patients
Y=df.iloc[:,0].values#Whether patient has cancer or not

#Train test split
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)#Scaling X_train
X_test=sc.fit_transform(X_test)#Scaling X_test

#Function for  different models
def models(X_train,Y_train):

    #Logistic regression
    from sklearn.linear_model import LogisticRegression
    log=LogisticRegression(random_state=0)
    log.fit(X_train,Y_train)

    #Decision tree
    from sklearn.tree import DecisionTreeClassifier
    tree=DecisionTreeClassifier(criterion='entropy',random_state=0)
    tree.fit(X_train,Y_train)

    #Random forest classifier
    from sklearn.ensemble import RandomForestClassifier
    forest=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
    forest.fit(X_train,Y_train)

    #GaussianNB
    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    gnb.fit(X_train,Y_train)

    #Printing accuracy
    print("Logistic regression:",log.score(X_train,Y_train))
    print("Decision Tree:",tree.score(X_train,Y_train))
    print("Random Forest:",forest.score(X_train,Y_train))
    print("GaussianNB:",gnb.score(X_train,Y_train))
    return log,tree,forest,gnb

#Testing Function for all models
print("Accuracy")
model=models(X_train,Y_train)

#Metrics of the models
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
for i in range(len(model)):
    print("\nModel",i+1)
    s=""
    if i==0:
        print("LOGISTIC REGRESSION: \n")
        s="Logistic Regression Algorithm"
    if i==1:
        print("USING DECISION TREES: \n")
        s="Decision Trees Algorithm"
    if i==2:
        print("USING RANDOM FORESTS: \n")
        s="Random Forest Algorithm"
    if i==3:
        print("USING GAUSSIAN NAIVE BAYES: \n")
        s="Gaussian Naive Bayes Algorithm"
        
    print("Classification Report:")
    print(classification_report(Y_test,model[i].predict(X_test)))
    print("Accuracy Score of the Model using",s,":",accuracy_score(Y_test,model[i].predict(X_test)))
    print("Accuracy Percentage of the Model using",s,":",100*accuracy_score(Y_test,model[i].predict(X_test)),"%")