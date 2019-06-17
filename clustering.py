import sklearn as sk
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from itertools import cycle, islice
import matplotlib.colors as mcolors
from sklearn.cluster import MiniBatchKMeans
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.metrics import balanced_accuracy_score, accuracy_score


def create_cluster_coalitions(models: list, data: pd.DataFrame, threshold: float = 0.3):
    coalitions = []
    print('creating coalitions')
    for model in models:
        probs = model.predict_proba(data)
        probs[probs < threshold] = 0
        con_mat = np.matmul(probs, np.transpose(probs))
        col = sk.cluster.AgglomerativeClustering(n_clusters=2, connectivity=con_mat).fit_predict(data)
        coalitions.append(col)
    return coalitions


def get_clustering(features: pd.DataFrame, labels: pd.Series):
    n_clusters = 3
    clustering_methods = [MiniBatchKMeans(n_clusters=n_clusters), BayesianGaussianMixture(n_components=n_clusters)]
    return learn_clusters(features, labels, clustering_methods)


def cluster_party(features_subset: pd.DataFrame, party: str, clustering_method):
    clusters = clustering_method.fit_predict(features_subset)
    return party + pd.Series(clusters, index=features_subset.index).astype(str)


def cluster_parties(features: pd.DataFrame, labels: pd.Series, clustering_method):
    parties_clusters = [cluster_party(features[labels == party], party, clustering_method) for party in labels.unique()]
    return pd.concat(parties_clusters).sort_index()


def learn_clusters(features: pd.DataFrame, labels: pd.Series, clustering_methods):
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
        print('!')
        clf = RandomForestClassifier(n_jobs=-1)
        clf.set_params(**params)

        clusters = cluster_parties(features, labels, method)

        unique_clusters = np.unique(clusters)
        print(f'clusters are {unique_clusters}, len={len(unique_clusters)}')

        print('fitting forest')
        clf.fit(features, clusters)
        pred = clf.predict(features)
        print(f'RandomForest has a balanced accuracy of {balanced_accuracy_score(clusters, pred)} on the clusters')
        clustering_classifiers.append(clf)
        print('')

    return clustering_classifiers


def show_clusters(features: pd.DataFrame, labels: pd.Series, clustering_clf):
    y_pred = pd.Series(clustering_clf.predict(features)).astype('category').cat.codes

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = np.append(prop_cycle.by_key()['color'], [])

    if len(colors) < len(y_pred.unique()):
        added_colors = np.random.choice(list(mcolors.XKCD_COLORS.values()), size=len(y_pred.unique()) - len(colors))
        colors = np.append(colors, added_colors)

    # colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
    #                                      '#f781bf', '#a65628', '#984ea3',
    #                                      '#999999', '#e41a1c', '#dede00']),
    #                               int(max(y_pred) + 1))))

    # add black color for outliers (if any)
    # colors = np.append(colors, ["#000000"])

    pca = PCA(n_components=2, svd_solver='randomized')
    X = pca.fit(features).transform(features)

    # lda = LinearDiscriminantAnalysis(n_components=2)
    # X = lda.fit_transform(features, labels)

    plt.title('Clusters at the PCA axis for 2-dim')
    plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred.values])

    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.xticks(())
    plt.yticks(())
    plt.show()
