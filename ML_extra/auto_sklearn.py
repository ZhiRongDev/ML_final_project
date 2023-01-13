import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn import metrics

import autosklearn.classification
import sklearn.model_selection
import sklearn.metrics

from pprint import pprint


data_train = pd.read_csv('./data/train_jan06_task4bonus.csv')
data_test = pd.read_csv('./data/test_jan06_task4bonus.csv')

X = data_train[data_train.columns[:-1]]
Y = data_train['class']


x_train, x_test, y_train, y_test = train_test_split(X, Y, stratify=Y ,test_size=0.25, random_state=1)

def summarize_classification(y_test, y_pred):
    num_acc = metrics.accuracy_score(y_test, y_pred, normalize=False)
    F1_score = metrics.f1_score(y_test, y_pred, average='macro')    

    print("Test data count: ", len(y_test))
    print("accuracy_count: ", num_acc)
    print("f1_score: ", F1_score)
    
def Predict_model(mode, x_train, y_train, x_test, y_test, data_test):
    start = time.time()
    print("================")
    print(f"{mode}")
    
    model = ''

    if mode == 'auto':
        model = autosklearn.classification.AutoSklearnClassifier(
            # time_left_for_this_task=600,
            # per_run_time_limit=40,
            resampling_strategy='cv',
            resampling_strategy_arguments={'folds': 5}
        )
        model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    summarize_classification(y_test, y_pred)

    print(model.leaderboard())
    print(model.show_models())

    print("\nAfter re-fit")
    model.refit(x_train, y_train)
    y_pred = model.predict(x_test)
    summarize_classification(y_test, y_pred)
     

    y_pred = model.predict(data_test)

    ## output file
    predict_ans = [ [index + 1, value] for (index, value) in enumerate(y_pred)] 
    with open('./auto_sk_60_submission.csv', 'w') as f:
        f.write('Id,Category\n')
        for i in predict_ans:
            f.write(f'{i[0]},{i[1]}\n')

    end = time.time()
    print(f'執行時間: {end - start} 秒\n')

Predict_model('auto', x_train, y_train, x_test, y_test, data_test)
