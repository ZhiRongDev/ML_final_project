import pandas as pd
import time

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score

from sklearn.preprocessing import LabelEncoder

###
# import numpy as np
from sklearn.impute import SimpleImputer
###

data_train = pd.read_csv('./train_dec10_task4_missing_supplement.csv')
data_test = pd.read_csv('./archive/test_dec08_task4_missing_only_features.csv')

le=LabelEncoder()
le.fit(data_train['class'])
data_train['class']=le.transform(data_train['class'])


###
my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(data_train))
imputed_X_train.columns = data_train.columns
data_train = imputed_X_train
###
# breakpoint()


def summarize_classification(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred, normalize=True)
    num_acc = accuracy_score(y_test, y_pred, normalize=False)

    prec = precision_score(y_test, y_pred, average = 'weighted')
    recall = recall_score(y_test, y_pred, average = 'weighted')

    print("Test data count: ", len(y_test))
    print("accuracy_count: ", num_acc)
    print("accuracy_score: ", acc)
    print("precision score: ", prec)
    print("recall score: ", recall)


def Predict_model(mode, x_train, y_train):
    start = time.time()
    
    print("================")
    print(f"{mode}")
    
    if mode == 'Logistic':
        model = LogisticRegression(max_iter=10000, class_weight='balanced').fit(x_train, y_train)
    elif mode == 'NaiveBayse':
        model = GaussianNB().fit(x_train, y_train)
    elif mode == 'DecisionTree':
        model = DecisionTreeRegressor().fit(x_train, y_train)
    elif mode == 'KNN':
        model = KNeighborsClassifier(n_neighbors=3).fit(x_train, y_train)
    elif mode == 'SVM':
        model = svm.SVC(class_weight='balanced').fit(x_train, y_train)
        # model = svm.SVC().fit(x_train, y_train)

    y_pred = model.predict(x_test)
    
    summarize_classification(y_test, y_pred)

    y_pred = y_pred.astype(int)  
    y_pred = le.inverse_transform(y_pred)
    
    # print(f'{y_pred=}')
    
    print(y_pred.size)
    
    end = time.time()
    print(f'執行時間: {end - start} 秒\n')

X = data_train[data_train.columns[:-1]]
Y = data_train['class']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

Predict_model('Logistic', x_train, y_train)
Predict_model('NaiveBayse', x_train, y_train)
Predict_model('DecisionTree', x_train, y_train)
Predict_model('KNN', x_train, y_train)
Predict_model('SVM', x_train, y_train)
