import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

print('Load datasets')
xTrain = np.load('/home/elena/repos/CVCS_project_DeepFake/xTrain.npy')
yTrain = np.load('/home/elena/repos/CVCS_project_DeepFake/yTrain.npy')
xTest = np.load('/home/elena/repos/CVCS_project_DeepFake/xTest.npy')
yTest = np.load('/home/elena/repos/CVCS_project_DeepFake/yTest.npy')
xVal = np.load('/home/elena/repos/CVCS_project_DeepFake/xVal.npy')
yVal = np.load('/home/elena/repos/CVCS_project_DeepFake/yVal.npy')
print('loading')
print('Train set of length', len(xTrain))
print('Number of features per sample', len(xTrain[0]))

assert len(xTrain)==len(yTrain)
print('-----------------SVM---------------------------------------')
print('Box Constraint Optimization')
C = [0.2, 0.5, 0.8, 1, 1.5, 3]
svm = {}
accuracy = []
for i, c in enumerate(C):
    svm[i] = SVC(C=c, kernel='rbf')
    svm[i].fit(xTrain, yTrain)
    pred = svm[i].predict(xVal)
    accuracy.append(accuracy_score(yVal, pred))



idx = accuracy.index(max(accuracy))
opt_svm = svm[idx]
pred_test = opt_svm.predict(xTest)
print('Final accuracy on the test set: ', accuracy_score(yTest, pred_test))

print('---------------------------------------------------------------------')
print('----------------------------Random Forest----------------------------------')
rf = {}
accuracy = []
L = [10, 100, 1000]
for i, l in enumerate(L):
    rf[i] = RandomForestClassifier(l)
    rf[i].fit(xTrain, yTrain)
    pred = rf[i].predict(xVal)
    accuracy.append(accuracy_score(yVal, pred))



idx = accuracy.index(max(accuracy))
opt_rf = rf[idx]
pred_test = opt_rf.predict(xTest)
print('Final accuracy on the test set: ', accuracy_score(yTest, pred_test))



"""TO DO LIST:
    - create also xVal, xTest and yVal, yTest
    - optimize svm hyperparameters
    - try with random forest model
    - check if the dataset is balanced!!!"""

