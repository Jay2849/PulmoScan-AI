FROM python:3.8-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -e .

# Pre-train and bake model weights into Docker container
RUN python3 main.py

ENV PORT=8080
EXPOSE 8080

CMD ["python3", "app.py"]