import sklearn as sk
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def learn_clusters(features: pd.DataFrame, labels: pd.DataFrame, clustering_methods):
    clustering_classifiers = []
    params = {
        'bootstrap': False,
        'class_weight': 'balanced',
        'criterion': 'gini',
        'max_depth': 20,
        'max_features': 'log2',
        'min_samples_leaf': 4,
        'min_samples_split': 10,
        'n_estimators': 1738,
        'warm_start': False
    }
    
    for method in clustering_methods:
        clf = RandomForestClassifier()
        clf.set_params(**params)
        
        clusters = method.fit_predict(features)
        
        clf.fit(features, clusters)
        clustering_classifiers.append(clf)
    
    return clustering_classifiers


def show_clusters(features: pd.DataFrame, labels: pd.DataFrame, clustering_clf):
    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(y_pred) + 1))))
        # add black color for outliers (if any)
    y_pred = clustering_clf.predict(features)
    colors = np.append(colors, ["#000000"])

    pca = PCA(n_components=2)
    X = pca.fit(features).transform(features)
    
    
    plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])

    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)
    plt.xticks(())
    plt.yticks(())
    plt.show()
