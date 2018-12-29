import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler,SMOTE, BorderlineSMOTE, ADASYN
from collections import Counter
from sklearn.feature_selection import SelectFromModel

Clinical = pd.read_csv('data_labels.csv')
GeneData = pd.read_csv('datasets.csv',header=None)
GeneData = GeneData[~np.all(GeneData == 0, axis=1)]
GeneTr= GeneData.T
Header = GeneTr.iloc[0]
GeneTr = GeneTr[1:]
GeneTr.columns = Header
CombinedData = pd.merge(GeneTr, Clinical, how="inner", left_on='ID', right_on='PATIENT_ID')


X = CombinedData.ix[:,1:60484]
y = CombinedData['LATERALITY']


clf1 = RandomForestClassifier(n_estimators=10)
clf1 = clf1.fit(X, y)
model = SelectFromModel(clf1, prefit=True)
X = model.transform(X)

# ros = RandomOverSampler(random_state=42)
# X, y = ros.fit_resample(X, y)


X, y = SMOTE(random_state=40).fit_resample(X, y)
#X, y = BorderlineSMOTE(random_state=49).fit_resample(X,y)

print(sorted(Counter(y).items()))

pca = PCA(n_components=3).fit(X)
pca_3d = pca.transform(X)

fig = plt.figure(1,figsize=(5,5))
plt.clf()
ax = Axes3D(fig,rect=[0,0,.95,1],elev=50,azim=134)

for i in range(len(y)):
    if(y[i] == 'Left'):
        left = ax.scatter(pca_3d[i][0], pca_3d[i][1],pca_3d[i][2], c='r')
    elif(y[i]=='Right'):
        right = ax.scatter(pca_3d[i][0], pca_3d[i][1],pca_3d[i][2], c='b' )
    else:
        bi = ax.scatter(pca_3d[i][0], pca_3d[i][1],pca_3d[i][2], c='g')

ax.set_xlabel('X - axis')
ax.set_ylabel('Y - axis')
ax.set_zlabel('Z - axis')
plt.title("SMOTE on Data")
plt.legend((left,right,bi),('LEFT','RIGHT','BILATERAL'))
plt.show()
