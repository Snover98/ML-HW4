import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from standartisation import DFScaler
from wrappers import *
from model_selection import target_features_split
from sklearn.svm import SVC
import pickle
from sklearn.ensemble import RandomForestClassifier
import clustering
from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics import davies_bouldin_score, completeness_score


def show_col_parties(coalition: pd.Series, labels: pd.Series):
    col_number = coalition.value_counts().idxmax()
    print('The coalition voters are:')
    print(labels[coalition == col_number].value_counts().sort_index())
    print('Out of:')
    print(labels.value_counts().sort_index())
    print(f'The coalition parties are {labels[coalition == col_number].unique()}')
    print(f'The Coalition has {100 * len(labels[coalition == col_number]) / len(coalition)}% of the votes')


def main():
    fitted = True
    data = pd.read_csv("full_train.csv")
    test = pd.read_csv("full_test.csv")

    test_X, test_y = target_features_split(test, "Vote")

    features = test.columns.values.tolist()
    selected_features = ["Avg_environmental_importance", "Avg_government_satisfaction", "Avg_education_importance",
                         "Avg_monthly_expense_on_pets_or_plants", "Avg_Residancy_Altitude", "Yearly_ExpensesK",
                         "Weighted_education_rank", "Number_of_valued_Kneset_members"]
    issues = [feat for feat in features if feat.startswith("Issue")]
    features = issues + selected_features + ["Vote"]

    data = data[features]
    test = test[features]

    changed_data = data.copy()

    scaler = DFScaler(data, selected_features)
    data = scaler.scale(data)
    test = scaler.scale(test)

    X, Y = target_features_split(data, "Vote")

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
    model = RandomForestClassifier(n_jobs=-1)
    model.set_params(**params)

    if not fitted:
        model.fit(X, Y)
        pickle.dump(model, open('fit_model.sav', 'wb'))
    else:
        model = pickle.load(open('fit_model.sav', 'rb'))

    # change the data
    changed_data["Avg_environmental_importance"] = changed_data["Avg_environmental_importance"] * 0.8
    changed_data["Avg_education_importance"] = changed_data["Avg_education_importance"] * 1.2  # less
    changed_data["Avg_monthly_expense_on_pets_or_plants"] = changed_data[
                                                                "Avg_monthly_expense_on_pets_or_plants"] * 0.4  # less
    changed_data["Yearly_ExpensesK"] = changed_data["Yearly_ExpensesK"] * 0.8
    changed_data["Number_of_valued_Kneset_members"] = changed_data["Number_of_valued_Kneset_members"] * 1  # no effect
    changed_data["Weighted_education_rank"] = changed_data["Weighted_education_rank"] * 1  # no effect
    changed_data["Avg_Residancy_Altitude"] = changed_data["Avg_Residancy_Altitude"] * 1  # no effect
    changed_data["Avg_Residancy_Altitude"] = changed_data["Avg_Residancy_Altitude"] * 1  # dont know hahahah

    changed_data = scaler.scale(changed_data)
    changed_x, _ = target_features_split(changed_data, "Vote")

    y_changed_pred = pd.Series(model.predict(changed_x))
    clustering_methods = [BayesianGaussianMixture(n_components=3)]
    cluster_models = clustering.learn_clusters(changed_x, y_changed_pred, clustering_methods)
    cluster_coalitions = clustering.create_cluster_coalitions(cluster_models, changed_x, y_changed_pred, threshold=0.1)

    col = pd.Series(Y.isin(cluster_coalitions[0]))
    show_col_parties(col.astype(np.int), y_changed_pred)
    print("ddddddd: ", davies_bouldin_score(changed_x, col))
    print("ccccccc: ", completeness_score(Y, col))


if __name__ == "__main__":
    main()
