import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel, SelectKBest
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from imblearn.over_sampling import RandomOverSampler, SMOTE
from collections import Counter
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import chi2
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
Clinical = pd.read_csv('data_labels.csv')
GeneData = pd.read_csv('new_dataset.csv', header=None)

GeneTr= GeneData.T
Header = GeneTr.iloc[1]
GeneTr = GeneTr[1:]
GeneTr.columns = Header
CombinedData = pd.merge(GeneTr, Clinical, how="inner", left_on='ID', right_on='PATIENT_ID')


X = CombinedData.ix[:,1:57465]
y = CombinedData['LATERALITY']
features = [10,20,30,40,50,60]
for est in features:
    print("ESTIMATOR:" + str(est))
    X_new = SelectKBest(chi2, k=est).fit_transform(X, y)
    print("Feature Selection shape" + str(X_new.shape))
    val = X_new.shape[1]


    ros = RandomOverSampler(random_state=0)
    X_res, y_res = ros.fit_resample(X_new, y)

    kf = KFold(n_splits=10, shuffle=True)
    #clf = GaussianNB()
    #clf = svm.SVC(kernel='linear', C=1.0)
    #clf = svm.SVC(kernel='rbf',gamma='auto')
    clf = KNeighborsClassifier(n_neighbors=22)

    avg = 0
    confusion_matrix_l = [[0,0,0],[0,0,0],[0,0,0]]
    #print("length of P" + str(len(P)))
    y_test = CombinedData['LATERALITY'][0:val+1]
    #print("LENGTH OF y_test =" + str(len(y_test)))

    for train_index, test_index in kf.split(X_res):
        X_train, X_test = X_res[train_index], X_res[test_index]
        y_train, y_test = y_res[train_index], y_res[test_index]
        clf.fit(X_train, y_train)
        P = clf.predict(X_test)
        confusion_matrix_l = confusion_matrix_l + confusion_matrix(y_test,P)
        avg = avg + accuracy_score(y_test, P)
    print(str((avg/10)*100)+ "%")
    print(confusion_matrix_l)