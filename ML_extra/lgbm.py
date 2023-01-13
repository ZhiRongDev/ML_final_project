import numpy as np
import pandas as pd
import time
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import lightgbm as lgb

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV
from imblearn.combine import SMOTETomek

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

data_train = pd.read_csv('./data/train_jan06_task4bonus.csv')
data_test = pd.read_csv('./data/test_jan06_task4bonus.csv')

## transform the "class" label into number
le=LabelEncoder()
le.fit(data_train['class'])
data_train['class']=le.transform(data_train['class'])

### Remove NA value
imp = IterativeImputer(random_state=0)
imp.fit(data_train)
data_train = pd.DataFrame(imp.transform(data_train), columns = data_train.columns) 
imp.fit(data_test)
data_test = pd.DataFrame(imp.transform(data_test), columns = data_test.columns)

# X = data_train[data_train.columns[:-1]]
Y = data_train['class']

features = ['feature0', 'feature6', 'feature5', 'feature4', 'feature2', 'feature3', 'feature1']
X = data_train[features]
data_test = data_test[features]

####  
smote_tomek = SMOTETomek()
X, Y = smote_tomek.fit_resample(X, Y)
####

x_train, x_test, y_train, y_test = train_test_split(X, Y, stratify=Y ,test_size=0.25, random_state=1)

def summarize_classification(y_test, y_pred):
    num_acc = metrics.accuracy_score(y_test, y_pred, normalize=False)
    F1_score = metrics.f1_score(y_test, y_pred, average='macro')
    print("Test data count: ", len(y_test))
    print("accuracy_count: ", num_acc)
    print("F1_score: ", F1_score)
    

def Predict_model(mode, x_train, y_train, x_test, y_test, data_test):
    start = time.time()
    print("================")
    print(f"{mode}")

    model = ''
    
    if mode == 'lgbm':        
        # model = lgb.LGBMClassifier(random_state=42, n_estimators=1000, max_depth=60, learning_rate=0.09, colsample_bytree=0.7)
        # model.fit(x_train, y_train)
        
        n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
        max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
        max_depth.append(None)
        learning_rate=[round(float(x),2) for x in np.linspace(start=0.01, stop=0.2, num=10)]
        colsample_bytree =[round(float(x),2) for x in np.linspace(start=0.1, stop=1, num=10)]

        random_grid = {'n_estimators': n_estimators,
               'max_depth': max_depth,
               'learning_rate': learning_rate,
               'colsample_bytree': colsample_bytree}
        clf = lgb.LGBMClassifier(random_state=42)
        grid = RandomizedSearchCV(estimator = clf, param_distributions=random_grid,
                              n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)

        model = grid.fit(x_train,y_train)

        ### Feature importance for top 13 predictors
        print(grid.best_params_)
        predictors = [x for x in x_train.columns]
        feat_imp = pd.Series(grid.best_estimator_.feature_importances_, predictors).sort_values(ascending=False)
        feat_imp = feat_imp[0:13]
        plt.rcParams['figure.figsize'] = 20, 5
        feat_imp.plot(kind='bar', title='Feature Importance')
        plt.ylabel('Feature Importance Score')
        plt.savefig("lgbm_noScale_iterative.png")
    
    y_pred = model.predict(x_test)
    summarize_classification(y_test, y_pred)

    y_pred = model.predict(data_test)

    ## transform back the "class" label in dataset
    y_pred = y_pred.astype(int)  
    y_pred = le.inverse_transform(y_pred)
    
    ## output file
    predict_ans = [ [index + 1, value] for (index, value) in enumerate(y_pred)] 
    with open('./lgbm_iterative_submission.csv', 'w') as f:
        f.write('Id,Category\n')
        for i in predict_ans:
            f.write(f'{i[0]},{i[1]}\n')

    end = time.time()
    print(f'執行時間: {end - start} 秒\n')

Predict_model('lgbm', x_train, y_train, x_test, y_test, data_test)
