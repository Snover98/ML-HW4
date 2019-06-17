import generative
import sklearn
import pandas as pd
import numpy as np
import clustering


def main():
    data = pd.read_csv("train_processed.csv")
    X, Y = generative.target_features_split(data, "Vote")

    # cluster models
    print('Doing clustering coalitions')
    cluster_models = clustering.get_clustering(X, Y)
    cluster_coalitions = clustering.create_cluster_coalitions(cluster_models, X, threshold=0.3)

    # generative_models
    print('Doing generative coalitions')
    gen_models = generative.train_generative(data)
    gen_coalitions = generative.create_gen_coalitions(gen_models, X, Y)

    for model in cluster_models + gen_models:
        clustering.show_clusters(X, model)

    # check how good the coalitions are
    scores = []
    for coalition in cluster_coalitions + gen_coalitions:
        scores.append(sklearn.metrics.davies_bouldin_score(X, coalition))
        print('')
        print('=========================================')
        print(f'Score is {scores[-1]}')
        clustering.show_labels(X, pd.Series(coalition))


if __name__ == '__main__':
    main()
