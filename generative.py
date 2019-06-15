import sklearn
import pandas as pd
from model_selection import target_features_split


def train_generative(data: pd.DataFrame):
    models = []
    models.append(sklearn.discriminant_analysis.LinearDiscriminantAnalysis())
    models.append(sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis())
    X, Y = target_features_split(data, "Vote")
    for model in models:
        model.fit(X, Y)
    return models
