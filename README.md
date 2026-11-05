# FormCheck

Приложение для оценки техники физических упражнений по фото и видео с помощью компьютерного зрения.

## Что уже есть

- CV-модуль оценки углов и выравнивания тела
- FastAPI backend для загрузки фото и видео
- Простая веб-страница для загрузки файла и просмотра результата
- Базовые тесты логики анализа

## Как запустить

Если проект уже лежит локально в папке, перейдите в корень репозитория и выполните:

```bash
cd /path/to/FormCheck
. vision/.venv/bin/activate
uvicorn server:app --host 0.0.0.0 --port 8000
```

Если вы находитесь прямо в папке проекта, достаточно:

```bash
. vision/.venv/bin/activate
uvicorn server:app --host 0.0.0.0 --port 8000
```

Откройте в браузере:

```text
http://127.0.0.1:8000/
```

## Какие упражнения поддерживаются

- squat
- push-up

## Основные точки проекта

- [server.py](server.py)
- [vision/analysis.py](vision/analysis.py)
- [vision/main.py](vision/main.py)
- [frontend/index.html](frontend/index.html)
- [frontend/app.js](frontend/app.js)
- [tests/test_pose_analysis.py](tests/test_pose_analysis.py)
