#Models
from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


class  ClassicModels:
  def __init__(self, name, model, params={}):
    self.name = name
    self.model = model
    self.params = params

def get_models_to_evaluate():
    knn = ClassicModels("KNN", KNeighborsClassifier)
    svc = ClassicModels("SVC", SVC)
    rf = ClassicModels("RF", RandomForestClassifier)
    return [knn, svc, rf]


def get_models_bayesian_to_evaluate():
    knn = ClassicModels("KNN",
                        KNeighborsClassifier,
                        {
                            'n_neighbors': {'type': 'int', 'low': 1, 'high': 30},
                            'weights': {'type': 'categorical', 'choices': ['uniform', 'distance']},
                            'p': {'type': 'int', 'low': 1, 'high': 2}
                        })
    svc = ClassicModels("SVC",
                        SVC,
                        {
                            'C': {'type': 'float', 'low': 1e-6, 'high': 1e2},
                            'kernel': {'type': 'categorical', 'choices': ['linear', 'poly', 'rbf', 'sigmoid']},
                            'degree': {'type': 'int', 'low': 2, 'high': 5},  # relevant for 'poly' kernel
                            'gamma': {'type': 'categorical', 'choices': ['scale', 'auto']},
                            'shrinking': {'type': 'boolean', 'choices': [True]},
                            "max_iter": {'type': 'int', 'low': 10000, 'high': 10000}
                        })
    rf = ClassicModels("RF",
                       RandomForestClassifier,
                       {
                            'n_estimators': {'type': 'int', 'low': 10, 'high': 200},
                            'max_depth': {'type': 'int', 'low': 2, 'high': 32},
                            'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
                            'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 20},
                            'max_features': {'type': 'categorical', 'choices': ['sqrt', 'log2']}
                       }
                      )
    return [knn, svc, rf]