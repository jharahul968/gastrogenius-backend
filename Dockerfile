FROM tiangolo/uvicorn-gunicorn:python3.9-slim

LABEL maintainer="jharahul968"

ENV WORKERS_PER_CORE=4
ENV MAX_WORKERS=24
ENV LOG_LEVEL="warning"
ENV TIMEOUT="200"

# Install system dependencies
RUN apt-get update \
    && apt-get install -y libgl1 libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
RUN mkdir /yolov5-fastapi
WORKDIR /yolov5-fastapi

# Copy and install Python dependencies
COPY requirements.txt /yolov5-fastapi
RUN pip install -r requirements.txt

# Copy the application code
COPY . /yolov5-fastapi

# Expose the application port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

