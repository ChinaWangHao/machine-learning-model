#!/usr/bin/env python3
"""
Use a trained spam detection model to classify emails.
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
import joblib

from spam_detector.preprocessing import preprocess_text
from spam_detector.config import load_config

def parse_args():
    parser = argparse.ArgumentParser(description='Classify emails using trained spam detection model')
    parser.add_argument('--config', type=str, default='config.toml', 
                        help='Path to the TOML configuration file')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to the trained model file')
    parser.add_argument('--vectorizer', type=str, required=True,
                        help='Path to the trained vectorizer file')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to the CSV file containing emails to classify')
    parser.add_argument('--text-col', type=str, default='text',
                        help='Name of the column containing email text')
    parser.add_argument('--output', type=str, default='predictions.csv',
                        help='Path to save prediction results')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Load model and vectorizer
        print("Loading model and vectorizer...")
        model = joblib.load(args.model)
        vectorizer = joblib.load(args.vectorizer)
        
        # Load data
        print("Loading data...")
        data = pd.read_csv(args.input)
        
        # Preprocess text
        print("Preprocessing text...")
        data['processed_text'] = data[args.text_col].apply(
            lambda x: preprocess_text(x, config['preprocessing'])
        )
        
        # Vectorize text
        print("Vectorizing text...")
        X = vectorizer.transform(data['processed_text'])
        
        # Make predictions
        print("Making predictions...")
        data['prediction'] = model.predict(X)
        
        # Add probability scores if available
        if hasattr(model, 'predict_proba'):
            data['spam_probability'] = model.predict_proba(X)[:, 1]
        
        # Save predictions
        print(f"Saving predictions to {args.output}...")
        result_columns = [args.text_col, 'prediction']
        if 'spam_probability' in data.columns:
            result_columns.append('spam_probability')
        
        data[result_columns].to_csv(args.output, index=False)
        
        # Print summary
        spam_count = data['prediction'].sum()
        ham_count = len(data) - spam_count
        print(f"\nClassification complete:")
        print(f"Total emails: {len(data)}")
        print(f"Spam emails: {spam_count} ({spam_count / len(data) * 100:.1f}%)")
        print(f"Ham emails: {ham_count} ({ham_count / len(data) * 100:.1f}%)")
        
        return 0
    
    except Exception as e:
        print(f"Error during classification: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
