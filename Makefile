.PHONY: setup test lint format data train predict docker clean

# Default configuration values
CONFIG_FILE ?= config.toml
DATA_FILE ?= data/sample_data.csv
OUTPUT_DIR ?= models
NUM_SAMPLES ?= 1000
SPAM_RATIO ?= 0.3

# Setup the project
setup:
	pip install -e .

# Generate sample data
data:
	mkdir -p data
	python generate_sample.py --output $(DATA_FILE) --num-samples $(NUM_SAMPLES) --spam-ratio $(SPAM_RATIO)

# Train the model
train:
	python train.py --config $(CONFIG_FILE) --data $(DATA_FILE) --output-dir $(OUTPUT_DIR)

# Make predictions using the trained model
predict:
	python predict.py --model $(OUTPUT_DIR)/spam_model.joblib --vectorizer $(OUTPUT_DIR)/vectorizer.joblib --input $(DATA_FILE) --output predictions.csv

# Run tests
test:
	pytest tests/ --cov=spam_detector

# Run linting
lint:
	flake8 spam_detector/ --count --select=E9,F63,F7,F82 --show-source --statistics
	black --check spam_detector/
	isort --check-only spam_detector/

# Format code
format:
	black spam_detector/ tests/
	isort spam_detector/ tests/

# Build Docker image
docker:
	docker build -t spam-detector:latest .

# Run the Docker container
docker-run:
	docker run --rm -v $(PWD)/data:/app/data -v $(PWD)/models:/app/models spam-detector:latest --data /app/data/$(notdir $(DATA_FILE)) --output-dir /app/models

# Clean up generated files
clean:
	rm -rf models/* data/* predictions.csv
	rm -rf __pycache__ spam_detector/__pycache__ tests/__pycache__
	rm -rf .pytest_cache .coverage
