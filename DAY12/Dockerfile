FROM python:3.11

WORKDIR /app

COPY . .

RUN pip install --default-timeout=1000 -r requirements.txt

EXPOSE 5000

CMD ["python", "app.py"]