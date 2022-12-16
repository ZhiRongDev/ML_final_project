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
from collections import Counter
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek

from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC

###
from sklearn.impute import SimpleImputer
###

data_train = pd.read_csv('./train_dec10_task4_missing_supplement.csv')
data_test = pd.read_csv('./archive/test_dec08_task4_missing_only_features.csv')

## transform the "class" label into number
le=LabelEncoder()
le.fit(data_train['class'])
data_train['class']=le.transform(data_train['class'])

###
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(data_train)
data_train = pd.DataFrame(imp.transform(data_train), columns = data_train.columns) 

imp.fit(data_test)
data_test = pd.DataFrame(imp.transform(data_test), columns = data_test.columns)
###

X = data_train[data_train.columns[:-1]]
Y = data_train['class']

print(sorted(Counter(Y).items()))
# smote_enn = SMOTEENN()
# X_resampled, Y_resampled = smote_enn.fit_resample(X, Y)

smote_tomek = SMOTETomek()
X_resampled, Y_resampled = smote_tomek.fit_resample(X, Y)
print(sorted(Counter(Y_resampled).items()))

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
        lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(x_train, y_train)
        tmp = SelectFromModel(lsvc)        
        x_train = x_train.loc[:, tmp.get_support()]
        x_test = x_test.loc[:, tmp.get_support()]
        data_test = data_test.loc[:, tmp.get_support()]

        n_neighbors = [i for i in range(1, 11)]
        parameters = {
            'n_neighbors': n_neighbors
        }
        k_nn = KNeighborsClassifier()
        model = GridSearchCV(k_nn, parameters).fit(x_train, y_train)
    
    elif mode == 'SVM':
        lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(x_train, y_train)
        tmp = SelectFromModel(lsvc)        
        x_train = x_train.loc[:, tmp.get_support()]
        x_test = x_test.loc[:, tmp.get_support()]
        data_test = data_test.loc[:, tmp.get_support()]

        parameters = {
            'kernel':['linear', 'poly', 'rbf', 'sigmoid'], 
            'C':[1, 10, 100]
        }
        svc = svm.SVC()
        model = GridSearchCV(svc, parameters).fit(x_train, y_train)


    elif mode == 'MLP':
        lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(x_train, y_train)
        tmp = SelectFromModel(lsvc)        
        x_train = x_train.loc[:, tmp.get_support()]
        x_test = x_test.loc[:, tmp.get_support()]
        data_test = data_test.loc[:, tmp.get_support()]

        parameters = {
            'activation': ['identity', 'logistic', 'tanh', 'relu'],
        }
        mlp = MLPClassifier(max_iter=10000)
        model = GridSearchCV(mlp, parameters).fit(x_train, y_train)

    y_pred = model.predict(data_test)
    # summarize_classification(y_test, y_pred)

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
Predict_model('MLP', x_train, y_train, x_test, y_test, data_test)
