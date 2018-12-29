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
print(X)
y = CombinedData['LATERALITY']
estimation = [1,2,3,4,5]
for est in estimation:
    print("ESTIMATOR:" + str(est))
    clf1 = RandomForestClassifier(n_estimators=est)
    clf1 = clf1.fit(X, y)
    model = SelectFromModel(clf1, prefit=True)
    X_new = model.transform(X)
    print(X_new.shape)

    feat_labels = X.columns[0:]

    '''
    feats = {}  # a dict to hold feature_name: feature_importance
    for feature, importance in zip(X_new.columns, clf1.feature_importances_):
        feats[feature] = importance  # add the name/value pair
    importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
    importances.sort_values(by='Gini-importance').plot(kind='bar', rot=45)
    plt.show()
    '''

    for feature in zip(feat_labels, clf1.feature_importances_):
         if (feature[1] != 0):
             print(feature)

    val = X_new.shape[1]
    #X_res, y_res = SMOTE().fit_resample(X_new, y)


    ros = RandomOverSampler(random_state=0)
    X_res, y_res = ros.fit_resample(X_new, y)

    kf = KFold(n_splits=10, shuffle=True)
    #clf = GaussianNB()
    #clf = svm.SVC(kernel='linear', C=1.0)
    clf = svm.SVC(kernel='rbf',gamma='auto')
    #clf = KNeighborsClassifier(n_neighbors=22)


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