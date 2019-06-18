import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from standartisation import DFScaler
from wrappers import *
from model_selection import target_features_split
from sklearn.svm import SVC
import pickle

if __name__ == "__main__":
    fitted = True
    data = pd.read_csv("full_train.csv")
    test = pd.read_csv("full_test.csv")

    test_X, test_y = target_features_split(test, "Vote")

    features = test.columns.values.tolist()
    selected_features = ["Avg_environmental_importance", "Avg_government_satisfaction", "Avg_education_importance",
                         "Avg_monthly_expense_on_pets_or_plants", "Avg_Residancy_Altitude", "Yearly_ExpensesK",
                         "Weighted_education_rank", "Number_of_valued_Kneset_members"]
    features = [feat for feat in features if feat.startswith("Issue")] + selected_features + ["Vote"]

    data = data[features]
    test = test[features]

    changed_data = data.copy()

    scaler = DFScaler(data, selected_features)
    data = scaler.scale(data)
    test = scaler.scale(test)

    X, Y = target_features_split(data, "Vote")
    hist_true = Y.value_counts().astype(float) / len(Y.index)


    if not fitted:
        model = ElectionsResultsWrapper(
            SVC(C=7.70625, class_weight='balanced', degree=5, gamma='auto', kernel='poly', probability=True,
                tol=0.33618))
        model.fit(X, Y)
        pickle.dump(model, open('fit_model.sav', 'wb'))
    else:
        model = pickle.load(open('fit_model.sav', 'rb'))



    feature = "Avg_Residancy_Altitude"
    values = np.random.randint(12, high=14, size=len(Y))
    changed_data[feature] = values.astype(np.float)

    scaled_changed_data = scaler.scale(changed_data.copy())
    scaled_changed_X, _ = target_features_split(scaled_changed_data, "Vote")

    hist = model.predict(scaled_changed_X)
    print(hist - hist_true)
