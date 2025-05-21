FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir build && \
    pip install --no-cache-dir .

# Copy application code
COPY config.toml ./
COPY train.py predict.py generate_sample.py ./
COPY spam_detector/ ./spam_detector/

# Create directories
RUN mkdir -p /app/data /app/models

# Generate sample data
RUN python generate_sample.py --output /app/data/sample_data.csv --num-samples 1000

# Set default command to train the model
ENTRYPOINT ["python", "train.py"]
CMD ["--data", "/app/data/sample_data.csv", "--output-dir", "/app/models"]
