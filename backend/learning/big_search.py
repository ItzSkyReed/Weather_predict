import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import json
import joblib
import os

from xgboost import XGBClassifier

data = pd.read_csv('formatted_data_for_learning.csv')

X = data.drop(columns="weather_code")  # Обучаем
y = data["weather_code"]  # Предсказываем
X_train_raw, X_test_raw, y_train_raw, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# Масштабируем
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)


models = {
    'HistGradientBoosting': (
        Pipeline([
            ('model', HistGradientBoostingClassifier(early_stopping=True, random_state=100))
        ]),
        {
            'model__learning_rate': [0.01, 0.05, 0.1],
            'model__max_iter': [100, 200, 300],
            'model__max_depth': [3, 5, 10, None],
            'model__min_samples_leaf': [20, 50, 100],
            'model__l2_regularization': [0.0, 1.0, 5.0],
            'model__max_leaf_nodes': [15, 31, 63]
        }
    ),

    'XGBoost': (
        Pipeline([
            ('model', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=100))
        ]),
        {
            'model__n_estimators': [100, 200],
            'model__learning_rate': [0.05, 0.1],
            'model__max_depth': [3, 5, 7],
            'model__subsample': [0.8, 1.0],
            'model__colsample_bytree': [0.8, 1.0]
        }
    ),
    "CatBoost": (
        Pipeline([
            ('model', CatBoostClassifier(random_state=100))
        ]),
        {
            'model__iterations': [200, 350, 500],
            'model__learning_rate': [0.03, 0.1],
            'model__depth': [4, 6, 8]
        }
    ),
    'Extra Forest': (
        Pipeline([
            ('model', ExtraTreesClassifier(class_weight="balanced_subsample", random_state=100))
        ]),
        {
            "model__criterion": ["gini", "entropy"],
            'model__n_estimators': [100, 200],
            'model__max_depth': [5, 10, 20, None],
            'model__min_samples_split': [2, 5, 10],
            'model__max_features': ['sqrt', 'log2', 0.2, 0.5]
        }
    ),
    'Random Forest': (
        Pipeline([
            ('model', RandomForestClassifier(class_weight="balanced_subsample", random_state=100))
        ]),
        {
            "model__criterion": ["gini", "entropy"],
            'model__n_estimators': [100, 200],
            'model__max_depth': [5, 10, 20, None],
            'model__min_samples_split': [2, 5, 10],
            'model__max_features': ['sqrt', 'log2', 0.2, 0.5]
        }
    ),
    'Bagging + KNN': (
        Pipeline([
            ('model', BaggingClassifier(estimator=KNeighborsClassifier()))
        ]),
        {
            'model__n_estimators': [10, 20, 30],
            'model__estimator__n_neighbors': [*range(8, 31, 3)],
            'model__estimator__weights': ['uniform', 'distance'],
            'model__estimator__metric': ['euclidean', 'manhattan'],
        }
    ),
}

KF = StratifiedKFold(shuffle=True, random_state=100)

os.makedirs("saved_models", exist_ok=True)
all_metrics = []

for name, (pipeline, param_grid) in models.items():
    print(f"Training and tuning: {name}")
    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        cv=KF,
        n_jobs=-1,
        scoring='f1_macro',
        n_iter=300,
        verbose=2,
        return_train_score=True,
        random_state=100
    )

    random_search.fit(X_train, y_train_raw)
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    y_pred_test = best_model.predict(X_test)

    # Отчёт
    report = classification_report(y_test, y_pred_test, zero_division=0)

    # Метрики
    metrics = {
        "Model Name": name,
        "Best Estimator": str(best_model),
        "Accuracy": accuracy_score(y_test, y_pred_test),
        "Precision": precision_score(y_test, y_pred_test, average='macro', zero_division=0),
        "Recall": recall_score(y_test, y_pred_test, average='macro'),
        "F1": f1_score(y_test, y_pred_test, average='macro')
    }
    all_metrics.append(metrics)

    # Сохраняем параметры
    with open(f"saved_models/{name}_best_params.json", "w", encoding='utf-8') as f:
        json.dump(best_params, f, indent=4)

    # Сохраняем отчёт
    with open(f"saved_models/{name}_report.txt", "w", encoding='utf-8') as f:
        f.write(report)

    # Сохраняем модель
    joblib.dump(best_model, f"saved_models/{name}_best_model.pkl")