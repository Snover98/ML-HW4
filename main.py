import generative
import sklearn
import pandas as pd
import numpy as np


def create_coalition(models: list, data: pd.DataFrame, threshold: int):
    coalitions = []
    for model in models:
        probs = model.predict_proba(data)
        probs[probs < threshold] = 0
        con_mat = np.matmul(probs, np.transpose(probs))
        clustring_alg = sklearn.cluster.AgglomerativeClustering(n_clusters=2, connectivity=con_mat).fit(data)
        coalitions.append(clustring_alg.labels_)
    return coalitions


def main():
    data = pd.read_csv("train_processed")
    X, Y = generative.target_features_split(data, "Vote")

    # asafs models
    asaf_models = []

    # generative_models
    gen_model = generative.train_generative(data)

    # can be updated later
    threshold = 0.3
    coalitions = create_coalition(asaf_models + gen_model, X, threshold)

    # check how good the coalitions are
    scores = []
    for coalition in coalitions:
        scores.append(sklearn.metrics.davies_bouldin_score(X, coalition))
