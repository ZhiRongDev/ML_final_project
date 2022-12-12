import numpy as np
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
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer

####
from collections import Counter
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
####


data_train = pd.read_csv('./train_dec10_task4_missing_supplement.csv')
data_test = pd.read_csv('./archive/test_dec08_task4_missing_only_features.csv')

le=LabelEncoder()
le.fit(data_train['class'])
data_train['class']=le.transform(data_train['class'])

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(data_train)
data_train = pd.DataFrame(imp.transform(data_train), columns = data_train.columns) 

imp.fit(data_test)
data_test = pd.DataFrame(imp.transform(data_test), columns = data_test.columns) 


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

def Predict_model(mode, x_train, y_train, x_test, y_test, data_test):
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
        lsvc = svm.SVC(kernel='linear', class_weight='balanced').fit(x_train, y_train)
        tmp = SelectFromModel(lsvc)        
        x_train = x_train.loc[:, tmp.get_support()]
        x_test = x_test.loc[:, tmp.get_support()]

        model = svm.SVC(kernel='linear', class_weight='balanced').fit(x_train, y_train)
    
    elif mode == 'MLP':
        lsvc = svm.SVC(kernel='linear', class_weight='balanced').fit(x_train, y_train)
        tmp = SelectFromModel(lsvc)        
        x_train = x_train.loc[:, tmp.get_support()]
        x_test = x_test.loc[:, tmp.get_support()]
        data_test = data_test.loc[:, tmp.get_support()]
        
        model = MLPClassifier(random_state=1, max_iter=300).fit(x_train, y_train)

    y_pred = model.predict(x_test)
    
    summarize_classification(y_test, y_pred)

    y_pred = y_pred.astype(int)  
    y_pred = le.inverse_transform(y_pred)
     
    # print(y_pred.size)
    predict_ans = [ [index + 1, value] for (index, value) in enumerate(y_pred)]
     
    with open('./submission.csv', 'w') as f:
        f.write('Id,Category\n')
        for i in predict_ans:
            f.write(f'{i[0]},{i[1]}\n')
    
    end = time.time()
    print(f'執行時間: {end - start} 秒\n')

X = data_train[data_train.columns[:-1]]
Y = data_train['class']

####
print(sorted(Counter(Y).items()))
smote_enn = SMOTEENN(random_state=0)
X_resampled, Y_resampled = smote_enn.fit_resample(X, Y)

smote_tomek = SMOTETomek(random_state=0)
X_resampled, Y_resampled = smote_tomek.fit_resample(X, Y)
print(sorted(Counter(Y_resampled).items()))

# breakpoint()
####

x_train, x_test, y_train, y_test = train_test_split(X_resampled, Y_resampled, test_size=0.2)

Predict_model('MLP', x_train, y_train, x_test, y_test, data_test)
