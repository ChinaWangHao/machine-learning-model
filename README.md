# Spam Email Detection Model

A complete machine learning workflow to train scikit-learn models for detecting spam emails.

## Overview

This project provides a configurable and modular workflow for training and deploying spam detection models. It uses scikit-learn for machine learning and TOML for configuration, making it easy to experiment with different preprocessing strategies and model architectures.

## Features

- **Modular Design**: Separate components for data loading, preprocessing, model training, and evaluation
- **Configurable**: All model parameters and preprocessing steps controlled via a single TOML config file
- **Multiple Model Support**: Choose from Naive Bayes, SVM, Random Forest, or Logistic Regression
- **Text Preprocessing**: Configurable text cleaning with options for stemming, stop words removal, etc.
- **Model Evaluation**: Comprehensive evaluation metrics and visualizations
- **Sample Data Generation**: Utility to generate synthetic data for testing

## Project Structure

```
├── config.toml                 # Main configuration file
├── pyproject.toml              # Project dependencies and metadata
├── train.py                    # Main script to train the model
├── predict.py                  # Script to make predictions using trained model
├── generate_sample.py          # Utility to generate sample data
├── spam_detector/              # Core package
│   ├── __init__.py
│   ├── config.py               # Configuration handling
│   ├── data.py                 # Data loading and splitting
│   ├── preprocessing.py        # Text preprocessing
│   ├── model.py                # Model creation
│   └── evaluation.py           # Model evaluation
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/machine-learning-model.git
   cd machine-learning-model
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e .
   ```

## Usage

### Generating Sample Data

If you don't have your own dataset, you can generate a synthetic dataset:

```bash
python generate_sample.py --output data/sample_data.csv --num-samples 1000 --spam-ratio 0.3
```

### Training a Model

Train a spam detection model with default parameters:

```bash
python train.py --data data/emails.csv --text-col text --label-col label --output-dir models
```

This will:
1. Load the dataset
2. Preprocess the text according to config.toml
3. Train the model (Naive Bayes by default)
4. Evaluate the model
5. Save the model and vectorizer to the specified output directory

### Making Predictions

Use a trained model to classify new emails:

```bash
python predict.py --model models/spam_model.joblib --vectorizer models/vectorizer.joblib --input new_emails.csv --output predictions.csv
```

### Customizing the Model

Edit `config.toml` to change model parameters, preprocessing steps, or to switch to a different model type. For example, to use SVM instead of Naive Bayes:

```toml
[model]
type = "svm"  # Options: "naive_bayes", "svm", "random_forest", "logistic_regression"
```

## Dataset Format

The expected format for input data is a CSV file with at least two columns:
- A column containing the email text (default name: "text")
- A column containing the label (0 for ham, 1 for spam) (default name: "label")

## License

MIT
