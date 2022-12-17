import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV

###
from collections import Counter
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
from xgboost import XGBClassifier
###

data_train = pd.read_csv('./train_dec04_task2.csv')
data_test = pd.read_csv('./test_dec04_task2_only_features.csv')

## transform the "class" label into number
le=LabelEncoder()
le.fit(data_train['class'])
data_train['class']=le.transform(data_train['class'])

X = data_train[data_train.columns[:-1]]
Y = data_train['class']


### data Standardization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = pd.DataFrame(scaler.transform(X), columns = X.columns)

scaler.fit(data_test)
data_test = pd.DataFrame(scaler.transform(data_test), columns = data_test.columns)
###

####  
print(sorted(Counter(Y).items()))
# smote_enn = SMOTEENN()
# X_resampled, Y_resampled = smote_enn.fit_resample(X, Y)

smote_tomek = SMOTETomek()
X_resampled, Y_resampled = smote_tomek.fit_resample(X, Y)
print(sorted(Counter(Y_resampled).items()))

# breakpoint()
####

## do not set the random_state, in order to make sure the datasplit is "random"
x_train, x_test, y_train, y_test = train_test_split(X_resampled, Y_resampled, stratify=Y_resampled ,test_size=0.2)
# x_train, x_test, y_train, y_test = train_test_split(X, Y, stratify=Y ,test_size=0.2)

def summarize_classification(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred, normalize=True)
    num_acc = accuracy_score(y_test, y_pred, normalize=False)
    prec = precision_score(y_test, y_pred, average = 'weighted')
    recall = recall_score(y_test, y_pred, average = 'macro')

    print("Test data count: ", len(y_test))
    print("accuracy_count: ", num_acc)
    print("accuracy_score: ", acc)
    print("precision score: ", prec)
    print("recall score: ", recall)
    


def Predict_model(mode, x_train, y_train, x_test, y_test, data_test):
    start = time.time()
    print("================")
    print(f"{mode}")

    model = ''

    if mode == 'KNN':
        tmp = [i for i in range(1, 11)]

        parameters = {
            'n_neighbors': tmp
        }
        k_nn = KNeighborsClassifier()
        model = GridSearchCV(k_nn, parameters).fit(x_train, y_train)
        print(f"Best parameters: {model.best_params_}")    
    
    elif mode == 'SVM':
        parameters = {
            'kernel':['linear', 'poly', 'rbf', 'sigmoid'], 
            'C':[1, 10, 100]
        }
        svc = svm.SVC()
        model = GridSearchCV(svc, parameters).fit(x_train, y_train)
        print(f"Best parameters: {model.best_params_}")    

    elif mode == 'MLP':
        parameters = {
            'activation': ['identity', 'logistic', 'tanh', 'relu'],
        }
        mlp = MLPClassifier(max_iter=10000)
        model = GridSearchCV(mlp, parameters).fit(x_train, y_train)
        print(f"Best parameters: {model.best_params_}")    

    elif mode == 'xgboost':
        parameters = { 
            'max_depth': [3,6,10],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [100, 500, 1000],
            'colsample_bytree': [0.3, 0.7]
        }

        # parameters = { 
        #     'n_estimators': [100],
        #     'colsample_bytree': [0.3]
        # }
        xgb = XGBClassifier() 
        model = GridSearchCV(xgb, parameters).fit(x_train, y_train)
        print(f"Best parameters: {model.best_params_}")    
    
    y_pred = model.predict(x_test)
    summarize_classification(y_test, y_pred)

    y_pred = model.predict(data_test)
    ## transform back the "class" label in dataset
    y_pred = y_pred.astype(int)  
    y_pred = le.inverse_transform(y_pred)
    
    ## output file
    predict_ans = [ [index + 1, value] for (index, value) in enumerate(y_pred)] 
    with open('./submission.csv', 'w') as f:
        f.write('Id,Category\n')
        for i in predict_ans:
            f.write(f'{i[0]},{i[1]}\n')

    end = time.time()
    print(f'執行時間: {end - start} 秒\n')

# Predict_model('KNN', x_train, y_train, x_test, y_test, data_test)
# Predict_model('SVM', x_train, y_train, x_test, y_test, data_test)
# Predict_model('MLP', x_train, y_train, x_test, y_test, data_test)
Predict_model('xgboost', x_train, y_train, x_test, y_test, data_test)
