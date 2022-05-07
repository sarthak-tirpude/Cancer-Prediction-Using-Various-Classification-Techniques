#Importing needed python modules
import numpy as np
import pandas as pd
import warnings as wr
#Ignoring warnings
from sklearn.exceptions import UndefinedMetricWarning
wr.filterwarnings("ignore", category=UndefinedMetricWarning)

#Loading data into dataframe(df)
df=pd.read_csv('Prostate_cancer_data.csv')

print(df.head(10))#Print all data of top 10 rows
print(df.shape)#Print the row and clumn count of the data
print(df.isna().sum())#Print all columns with empty data along with sum of empty data

df=df.dropna(axis=1)#Drop the column with empty data
df=df.drop(['id'],axis=1)

#Encoding first column
from sklearn.preprocessing import LabelEncoder
labelencoder_X=LabelEncoder()#Calling LabelEncoder
df.iloc[:,0]=labelencoder_X.fit_transform(df.iloc[:,0].values)#Encoding the values of diagnosis column to values

#Splitting data for dependence
X=df.iloc[:,1:].values#Features of cancerous and non cancerous patients
Y=df.iloc[:,0].values#Whether patient has cancer or not

#Train-Test split
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=1)

#Standard scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)#Scaling X_train
X_test=sc.fit_transform(X_test)#Scaling X_test

#Importing algorithm libraries
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

#Function for  different models
def models(X_train,Y_train):

    #Logistic regression
    log=LogisticRegression(random_state=0)
    log.fit(X_train,Y_train)

    #Decision tree
    tree=DecisionTreeClassifier(criterion='entropy',random_state=0)
    tree.fit(X_train,Y_train)
    plot_tree(tree)

    #Random forest classifier
    forest=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
    forest.fit(X_train,Y_train)

    #GaussianNB
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
    