# ANN

# Part 1-Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing Dataset
dataset=pd.read_csv("Churn_Modelling.csv")
X= dataset.iloc[:,3:13].values
y=dataset.iloc[:,13].values

# Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le_x1=LabelEncoder()
le_x2=LabelEncoder()
X[:,1]=le_x1.fit_transform(X[:,1])
X[:,2]=le_x2.fit_transform(X[:,2])
ohe_x=OneHotEncoder(categorical_features=[1])
X=ohe_x.fit_transform(X).toarray()
le_y=LabelEncoder()
y=le_y.fit_transform(y)
X=X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2- Creating ANN

# Importing Keras Library and packages
from keras.models import Sequential
from keras.layers import Dense,Dropout

# Initializing ANN and adding Output, Hidden Layers
classifier=Sequential()
classifier.add(Dense(8,activation='relu',input_shape=(11,)))
classifier.add(Dropout(rate=0.1))
classifier.add(Dense(64,activation='relu'))
classifier.add(Dropout(rate=0.1))
classifier.add(Dense(128,activation="relu"))
classifier.add(Dropout(rate=0.1))
classifier.add(Dense(1,activation='sigmoid'))

# Compiling ANN
classifier.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
classifier.summary()

# Training ANN
classifier.fit(X_train,y_train,epochs=500,batch_size=10)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred=y_pred>0.5

# New Prediction on Single test set
new_pred=classifier.predict(sc.transform(np.array([[0,0,600,1,40,3,60000.0,2,1,1,50000]]))) 

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
print(confusion_matrix(y_test,y_pred))
print("accuracy =", accuracy_score(y_test, y_pred))
print(classification_report(y_test,y_pred))

# Evaluating, Improving and Tuning the ANN
# Evaluating
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
def BuildClassifier():
    classifier=Sequential()
    classifier.add(Dense(8,activation='relu',input_shape=(11,)))
    classifier.add(Dropout(rate=0.1))
    classifier.add(Dense(64,activation='relu'))
    classifier.add(Dropout(rate=0.1))
    classifier.add(Dense(128,activation="relu"))
    classifier.add(Dropout(rate=0.1))
    classifier.add(Dense(1,activation='sigmoid'))
    classifier.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
    return classifier
eval_classifier=KerasClassifier(build_fn=BuildClassifier,batch_size=10,nb_epoch=500)
accuracies=cross_val_score(eval_classifier,X_train,y_train,cv=10)
accuracies.mean()
accuracies.std()

# Improving 
from sklearn.model_selection import GridSearchCV
def BuildClassifier(opt,num):
    classifier=Sequential()
    classifier.add(Dense(6,activation='relu',input_shape=(11,)))
    classifier.add(Dense(num,activation='relu'))
    classifier.add(Dense(1,activation='sigmoid'))
    classifier.compile(optimizer=opt,loss='binary_crossentropy',metrics=['accuracy'])
    return classifier
eval_classifier=KerasClassifier(build_fn=BuildClassifier)
parameters={"batch_size":[10,25,32],
            "nb_epoch":[100,200,500],
            "optimizer":["adam","rmsprop"],
            "num":[11,32,64,128]}
grid_search=GridSearchCV(eval_classifier,param_grid=parameters,scoring='accuracy',cv=10)
grid_search=grid_search.fit(X_train,y_train)
best_param=grid_search.best_params_
best_accuracy=grid_search.best_score_
