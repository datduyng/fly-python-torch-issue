# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment varibles
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update \
  && apt-get install -y --no-install-recommends gcc \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# Copy project
COPY . /app

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port uWSGI will listen on
EXPOSE 8000

# Tell Docker our app is listen on port 8000 at runtime
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
