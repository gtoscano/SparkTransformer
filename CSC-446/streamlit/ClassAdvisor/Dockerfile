FROM python:3.11-slim

# (Optional) OS deps for common Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential wget && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Pipenv and lock dependencies
COPY Pipfile Pipfile.lock ./
RUN pip install --no-cache-dir pipenv \
  && PIPENV_VENV_IN_PROJECT=1 pipenv install --deploy --system

# Copy the rest of your app
COPY . .

# Streamlit defaults
EXPOSE 8501
ENV STREAMLIT_SERVER_PORT=8501 \
  STREAMLIT_BROWSER_SERVER_ADDRESS=0.0.0.0

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
