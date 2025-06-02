import pandas as pd
from catboost import CatBoostClassifier
from joblib import load
from sklearn.preprocessing import StandardScaler

model: CatBoostClassifier = load("app/models/weather_predict/CatBoost.pkl")
scaler = StandardScaler()

weather_map = {
    0: "Солнечно",
    1: "Малая Облачность",
    2: "Переменная Облачность",
    3: "Облачно",
    4: "Морось",
    5: "Дождь",
    6: "Снег"
}

def predict(data: pd.DataFrame) -> str:
    data_scaled = scaler.fit_transform(data.drop(columns=["date"]))
    # классы для каждой строки
    predictions = model.predict(data_scaled).ravel() #ravel делает список плоским.
    predictions = [weather_map[pred] for pred in predictions] # Преобразовываем числовые значения в читаемые,

    df = pd.DataFrame({
        "date": data["date"], # добавляем дату
        "class": predictions
    })

    return df.to_json(orient="records", index=False)