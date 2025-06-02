from fastapi import FastAPI, Request
from starlette.status import HTTP_411_LENGTH_REQUIRED, HTTP_400_BAD_REQUEST, HTTP_413_REQUEST_ENTITY_TOO_LARGE

from app.api import predict_weather
from fastapi.responses import ORJSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB

app = FastAPI(default_response_class=ORJSONResponse)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:63342"],  # Источник, с которого разрешены запросы
    allow_credentials=True,
    allow_methods=["*"],  # Разрешенные методы (например, GET, POST и т.д.)
    allow_headers=["*"],  # Разрешенные заголовки
)


class MaxUploadSizeMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Проверяем только POST-запросы с multipart-данными (файлы)
        if request.method == "POST" and request.headers.get("Content-Type", "").startswith("multipart/form-data"):

            # Пытаемся получить Content-Length
            content_length_header = request.headers.get("Content-Length")

            if content_length_header is None:
                return JSONResponse({"detail": "Заголовок Content-Length обязателен"}, status_code=HTTP_411_LENGTH_REQUIRED)

            try:
                content_length = int(content_length_header)
            except ValueError:
                return JSONResponse({"detail": "Неверный заголовок Content-Length"}, status_code=HTTP_400_BAD_REQUEST)

            if content_length > MAX_UPLOAD_SIZE:
                return JSONResponse({"detail": "Файл слишком большой"}, status_code=HTTP_413_REQUEST_ENTITY_TOO_LARGE)

        response = await call_next(request)
        return response


app.include_router(predict_weather.router, prefix="/api")
