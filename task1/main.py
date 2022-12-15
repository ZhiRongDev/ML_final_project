import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

from sklearn.preprocessing import LabelEncoder

###
from sklearn.model_selection import GridSearchCV
###

data_train = pd.read_csv('train_nov28_task1.csv')
data_test = pd.read_csv('test_nov28_task1_only_features.csv')

## transform the "class" label into number
le=LabelEncoder()
le.fit(data_train['class'])
data_train['class']=le.transform(data_train['class'])

X = data_train[data_train.columns[:-1]]
Y = data_train['class']

## do not set the random_state, in order to make sure the datasplit is "random"
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

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
    
    ## max_iter set to 1000 in this case is just enough to convergent to optimal result
    if mode == 'Logistic':
        model = LogisticRegression(max_iter=1000).fit(x_train, y_train)
    elif mode == 'NaiveBayse':
        model = GaussianNB().fit(x_train, y_train)
    elif mode == 'DecisionTree':
        model = DecisionTreeRegressor().fit(x_train, y_train)
    
    elif mode == 'KNN':
        parameters = {
            'n_neighbors':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        }
        k_nn = KNeighborsClassifier()
        model = GridSearchCV(k_nn, parameters).fit(x_train, y_train)
    
    elif mode == 'SVM':
        parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
        svc = svm.SVC()
        model = GridSearchCV(svc, parameters).fit(x_train, y_train)

    elif mode == 'MLP':
        parameters = {
            'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
            'activation': ['tanh', 'relu'],
            'solver': ['sgd', 'adam'],
            'alpha': [0.0001, 0.05],
            'learning_rate': ['constant','adaptive'],
        }
        mlp = MLPClassifier(max_iter=1000)
        model = GridSearchCV(mlp, parameters).fit(x_train, y_train)
        
    y_pred = model.predict(x_test)
    
    summarize_classification(y_test, y_pred)

    ## transform back the "class" label in dataset
    y_pred = y_pred.astype(int)  
    y_pred = le.inverse_transform(y_pred)
    
    ## output file
    # predict_ans = [ [index + 1, value] for (index, value) in enumerate(y_pred)] 
    # with open('./submission.csv', 'w') as f:
    #     f.write('Id,Category\n')
    #     for i in predict_ans:
    #         f.write(f'{i[0]},{i[1]}\n')

    end = time.time()
    print(f'執行時間: {end - start} 秒\n')

# Predict_model('Logistic', x_train, y_train, x_test, y_test, data_test)
# Predict_model('NaiveBayse', x_train, y_train, x_test, y_test, data_test)
# Predict_model('DecisionTree', x_train, y_train, x_test, y_test, data_test)
# Predict_model('KNN', x_train, y_train, x_test, y_test, data_test)
# Predict_model('SVM', x_train, y_train, x_test, y_test, data_test)
Predict_model('MLP', x_train, y_train, x_test, y_test, data_test)
