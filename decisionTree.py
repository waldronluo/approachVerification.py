import os
import numpy

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def decisionTree(X, y):
    clf = DecisionTreeClassifier().fit(X, y)
    print(clf.predict_proba(X))
    print(y)
    print(numpy.subtract(y, clf.predict(X)))

def randomForest(X, y):
    rdf = RandomForestClassifier(criterion=r'gini').fit(X, y)
    print(rdf.predict_proba(X))
    print(y)
    print(numpy.subtract(y, rdf.predict(X)))


if __name__ == '__main__':
    featurePath = "./buildData/"
    dirs = os.listdir(featurePath)

    X = []
    Y = []
    for dir in dirs:
        if(dir.startswith(r'features') and dir.endswith('npz')):
            dataPath = os.path.join(featurePath, dir)
            datfile = numpy.load(dataPath)
            features = datfile['features']
            y = datfile['y']
            X.append(features)
            Y.append(y)

    randomForest(X, Y)





