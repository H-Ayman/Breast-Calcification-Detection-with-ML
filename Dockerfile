FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8 
FROM python:3.9

WORKDIR /app

COPY mlops3/requirements.txt .


RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY mlops3 .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

