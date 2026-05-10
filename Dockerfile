FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    libpq-dev \
    libmariadb-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Setup environment variables
ENV OPENAI_API_KEY=""

# Default to serving the Frappe API
CMD ["python", "-m", "frappe.utils.bench_helper", "frappe", "serve", "--port", "8000"]
