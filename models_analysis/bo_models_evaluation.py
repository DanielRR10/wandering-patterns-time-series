import optuna
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import numpy as np

# Define a function to suggest hyperparameters based on the provided model definition
def suggest_hyperparameters(trial, model_def):
    hyperparameters = {}
    for param_name, param_options in model_def.items():
        if param_options['type'] == 'int':
            hyperparameters[param_name] = trial.suggest_int(param_name, param_options['low'], param_options['high'])
        elif param_options['type'] == 'float':
            hyperparameters[param_name] = trial.suggest_float(param_name, param_options['low'], param_options['high'])
        elif param_options['type'] == 'categorical':
            hyperparameters[param_name] = trial.suggest_categorical(param_name, param_options['choices'])
        elif param_options['type'] == 'boolean':
            hyperparameters[param_name] = trial.suggest_categorical(param_name, param_options['choices'])
    return hyperparameters

# Define the objective function for Optuna
def objective(trial, model_class, model_def, X_train, y_train):
    # Suggest hyperparameters
    hyperparameters = suggest_hyperparameters(trial, model_def)
    
    # Create the model with the suggested hyperparameters
    try:
        model = model_class(probability=True, **hyperparameters)
    except Exception as err:
        model = model_class(**hyperparameters)

        
    
    # Use a pipeline to scale the data
    pipeline = Pipeline([
        ('model', model)
    ])
    
    # Perform cross-validation and calculate the required metrics
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    roc_auc_scores = []
    for train_index, val_index in skf.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
        
        
        pipeline.fit(X_train_fold, y_train_fold)
        y_pred = pipeline.predict(X_val_fold)
        y_proba = pipeline.predict_proba(X_val_fold)
        
        accuracy_scores.append(pipeline.score(X_val_fold, y_val_fold))
        precision_scores.append(precision_score(y_val_fold, y_pred, average='macro'))
        recall_scores.append(recall_score(y_val_fold, y_pred, average='macro'))
        f1_scores.append(f1_score(y_val_fold, y_pred, average='macro'))
        roc_auc_scores.append(roc_auc_score(y_val_fold, y_proba, multi_class='ovr', average='macro'))
    
    # Calculate the mean of each metric
    mean_accuracy = np.mean(accuracy_scores)
    mean_precision = np.mean(precision_scores)
    mean_recall = np.mean(recall_scores)
    mean_f1 = np.mean(f1_scores)
    mean_roc_auc = np.mean(roc_auc_scores)
    
    # Store metrics in the trial
    trial.set_user_attr("accuracy", mean_accuracy)
    trial.set_user_attr("precision_macro", mean_precision)
    trial.set_user_attr("recall_macro", mean_recall)
    trial.set_user_attr("f1_macro", mean_f1)
    trial.set_user_attr("roc_auc_ovr_macro", mean_roc_auc)
    
    # Use accuracy as the optimization target
    return mean_accuracy