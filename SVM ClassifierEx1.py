# Support Vector Machine(SVM) Classification

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv(r'C:\Users\LENOVO\OneDrive\Desktop\fsds materials\fsds\3. Aug\29th\Social_Network_Ads.csv')
X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.svm import SVC
classifier = SVC()
#classifier = SVC(kernel='poly',gamma='scale',degree=3)
#classifier = SVC(kernel='poly',gamma='auto',degree=3)
#classifier = SVC(kernel='poly',gamma='scale',degree=4)
#classifier = SVC(kernel='poly',gamma='auto',degree=4)
#classifier = SVC(kernel='sigmoid',gamma='scale',degree=3)
#classifier = SVC(kernel='sigmoid',gamma='auto',degree=3)
#classifier = SVC(kernel='sigmoid',gamma='auto',degree=4)
#classifier = SVC(kernel='sigmoid',gamma='auto',degree=4)
#classifier = SVC(kernel='linear')

classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)
y_pred

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test,y_pred)
ac

from sklearn.metrics import classification_report
cr = classification_report(y_test,y_pred)
cr

bias = classifier.score(X_train,y_train)
bias

variance = classifier.score(X_test,y_test)
variance

# visualization
'''
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()'''

# future records

dataset1 = pd.read_csv(r'C:\Users\LENOVO\OneDrive\Desktop\fsds materials\fsds\3. Aug\29th\Future prediction1.csv')
d2 = dataset1.copy()
dataset1 = dataset1.iloc[:,[2,3]].values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
M = sc.fit_transform(dataset1)
M

y_pred1 = pd.DataFrame(M)

d2['y_pred1'] = classifier.predict(M)

d2.to_csv('Final1.csv')
import os
os.getcwd()

