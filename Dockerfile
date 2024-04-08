FROM python:3.10-slim-bookworm
# Install system dependencies
RUN apt-get update \
    && apt-get install -y libgl1 libglib2.0-0 \
    && apt-get install gcc -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
RUN mkdir /yolov5-model
WORKDIR /yolov5-model

# Copy and install Python dependencies
COPY requirements.txt /yolov5-model
RUN pip install numpy \
    && pip install Cython \
    && pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . /yolov5-model

# Expose the application port
EXPOSE 8000

# Run the application
CMD ["python3","main.py"]

