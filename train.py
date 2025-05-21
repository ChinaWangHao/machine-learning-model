#!/usr/bin/env python3
"""
Train a spam email detection model based on the configuration in config.toml.
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import joblib

from spam_detector.config import load_config
from spam_detector.data import load_data, split_data
from spam_detector.preprocessing import preprocess_text, create_vectorizer
from spam_detector.model import create_model
from spam_detector.evaluation import evaluate_model, plot_confusion_matrix

def parse_args():
    parser = argparse.ArgumentParser(description='Train spam email detection model')
    parser.add_argument('--config', type=str, default='config.toml', 
                        help='Path to the TOML configuration file')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to the CSV file containing email data')
    parser.add_argument('--text-col', type=str, default='text',
                        help='Name of the column containing email text')
    parser.add_argument('--label-col', type=str, default='label',
                        help='Name of the column containing spam/ham labels (1 for spam, 0 for ham)')
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Directory to save model artifacts')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        return 1
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and split data
    try:
        print("Loading data...")
        data = load_data(args.data)
        
        # Display dataset info
        print(f"Dataset shape: {data.shape}")
        print(f"Column names: {', '.join(data.columns)}")
        
        # Preprocess each text entry according to config
        print("Preprocessing text data...")
        data['processed_text'] = data[args.text_col].apply(lambda x: preprocess_text(x, config['preprocessing']))
        
        # Split data
        print("Splitting data into train and test sets...")
        X_train, X_test, y_train, y_test = split_data(
            data, 
            'processed_text', 
            args.label_col, 
            config['data']['train_test_split'], 
            config['data']['random_state']
        )
        
        print(f"Training set size: {len(X_train)}")
        print(f"Testing set size: {len(X_test)}")
        
        # Create and fit the vectorizer
        print("Vectorizing text data...")
        vectorizer = create_vectorizer(config['preprocessing'])
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        print(f"Number of features: {X_train_vec.shape[1]}")
        
        # Create and train the model
        print(f"Training {config['model']['type']} model...")
        model = create_model(config)
        model.fit(X_train_vec, y_train)
        
        # Evaluate the model
        print("Evaluating model...")
        evaluation_results = evaluate_model(
            model, X_train_vec, X_test_vec, y_train, y_test, config
        )
        
        # Print evaluation results
        print("\nModel Evaluation Results:")
        for metric, value in evaluation_results.items():
            print(f"{metric}: {value:.4f}")
        
        # Print detailed classification report
        y_pred = model.predict(X_test_vec)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Plot confusion matrix
        cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
        plot_confusion_matrix(y_test, y_pred, cm_path)
        print(f"Confusion matrix saved to {cm_path}")
        
        # Save the model and vectorizer
        model_path = os.path.join(args.output_dir, 'spam_model.joblib')
        vectorizer_path = os.path.join(args.output_dir, 'vectorizer.joblib')
        
        joblib.dump(model, model_path)
        joblib.dump(vectorizer, vectorizer_path)
        
        print(f"Model saved to {model_path}")
        print(f"Vectorizer saved to {vectorizer_path}")
        
        return 0
    
    except Exception as e:
        print(f"Error during model training: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
