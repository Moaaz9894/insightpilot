FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV INSIGHTPILOT_HOME=/app/store PYTHONUNBUFFERED=1
# Bake data + warehouse + trained model into the image: fast cold starts, reproducible builds.
# Optional real data: baked in if store/favorita/raw/sales_raw.csv exists locally, otherwise synthetic data is generated.
COPY Dockerfile store/favorita/raw/sales_raw.cs[v] /tmp/data/
RUN mkdir -p store/raw && if [ -f /tmp/data/sales_raw.csv ]; then mv /tmp/data/sales_raw.csv store/raw/sales_raw.csv; fi && rm -rf /tmp/data
RUN python -m insightpilot.bootstrap
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request as u;u.urlopen('http://localhost:8000/health')" || exit 1
CMD ["uvicorn", "insightpilot.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
