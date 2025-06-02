import io

import pandas as pd
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services import predict_weather

router = APIRouter()

# Название необходимых колонок с https://open-meteo.com/en/docs/historical-weather-api
REQUIRED_COLUMNS = ["date", "precipitation_sum", "precipitation_hours", "rain_sum", "snowfall_sum", "cloud_cover_mean", "cloud_cover_max", "cloud_cover_min", "temperature_2m_mean",
                    "temperature_2m_max", "temperature_2m_min", "apparent_temperature_mean", "apparent_temperature_min", "apparent_temperature_max", "wind_direction_10m_dominant",
                    "wind_gusts_10m_max", "wind_speed_10m_max", "relative_humidity_2m_mean", "relative_humidity_2m_max", "relative_humidity_2m_min", "shortwave_radiation_sum",
                    "dew_point_2m_mean", "dew_point_2m_max", "dew_point_2m_min", "sunshine_duration", "surface_pressure_mean"]
NUMERIC_COLUMNS = REQUIRED_COLUMNS.copy()
NUMERIC_COLUMNS.remove("date")


@router.post("/predict_weather/",
             responses={
                 400: {"description": "Неверные данные: формат, структура или содержание CSV"},
                 422: {"description": "Ошибка валидации запроса"},
                 500: {"description": "Внутренняя ошибка сервера"},
             })
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    if not file.filename.lower().endswith('.csv'):
        raise HTTPException(status_code=400, detail="Формат файла не CSV")

    try:
        user_weather_params = pd.read_csv(io.StringIO(contents.decode()))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Не удалось прочитать CSV: {str(e)}")

    user_weather_params.columns = user_weather_params.columns.str.strip()

    missing = [col for col in REQUIRED_COLUMNS if col not in user_weather_params.columns]

    if missing:
        raise HTTPException(status_code=400, detail=f"Отсутствуют необходимые столбцы: {', '.join(missing)}")


    if user_weather_params.empty:
        raise HTTPException(status_code=400, detail="Файл не содержит данных")

    if user_weather_params[REQUIRED_COLUMNS].isnull().any().any():
        raise HTTPException(status_code=400, detail="Обнаружены пустые значения в данных")

    for col in NUMERIC_COLUMNS:
        if not pd.api.types.is_numeric_dtype(user_weather_params[col]):
            raise HTTPException(status_code=400, detail=f"Столбец '{col}' должен быть числовым")

    # Мы меняем порядок столбцов на ожидаемый CatBoost моделью
    user_weather_params = user_weather_params[REQUIRED_COLUMNS]

    results: str = predict_weather.predict(user_weather_params)

    return results
