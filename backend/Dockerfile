FROM python:3.13-slim 

# Prevents Python from writing pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory 
WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/* 

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies 
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code 
COPY backend /app/backend 

# Expose FastAPI port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "backend.app.main_api:app", "--host", "0.0.0.0", "--port", "8000"]