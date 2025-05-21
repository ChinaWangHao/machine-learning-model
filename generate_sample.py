#!/usr/bin/env python3
"""
Generate a sample dataset for spam email detection.
This is useful for testing the workflow without real data.
"""
import argparse
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_20newsgroups

def parse_args():
    parser = argparse.ArgumentParser(description='Generate sample dataset for spam detection')
    parser.add_argument('--output', type=str, default='sample_data.csv',
                        help='Path to save the generated dataset')
    parser.add_argument('--num-samples', type=int, default=1000,
                        help='Number of samples to generate')
    parser.add_argument('--spam-ratio', type=float, default=0.3,
                        help='Ratio of spam emails in the dataset')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    print(f"Generating sample dataset with {args.num_samples} emails ({args.spam_ratio*100:.1f}% spam)...")
    
    # Calculate number of spam and ham emails
    num_spam = int(args.num_samples * args.spam_ratio)
    num_ham = args.num_samples - num_spam
    
    # Load 20 newsgroups dataset for realistic text content
    categories = [
        'comp.sys.mac.hardware', 'comp.windows.x', 'rec.autos', 'rec.motorcycles',
        'sci.crypt', 'sci.space', 'talk.politics.misc', 'talk.religion.misc'
    ]
    
    print("Fetching text data from 20 newsgroups dataset...")
    newsgroups = fetch_20newsgroups(
        subset='all',
        categories=categories,
        remove=('headers', 'footers', 'quotes')
    )
    
    # Randomly select posts for ham emails
    ham_indices = np.random.choice(len(newsgroups.data), num_ham, replace=False)
    ham_texts = [newsgroups.data[i] for i in ham_indices]
    
    # Generate synthetic spam emails
    spam_phrases = [
        "Congratulations! You've won", "FREE", "Limited time offer", "Act now", 
        "Don't miss this opportunity", "URGENT", "Make money fast", "Cash prize",
        "Low-cost insurance", "Discount", "Special offer", "Click here", "Subscribe now",
        "Amazing deals", "Exclusive access", "Best rates", "Guaranteed", "Risk-free",
        "Buy now", "Sale", "50% off", "Incredible deal", "Apply today", "Get rich",
        "Work from home", "Easy money", "Earn extra income", "No experience needed"
    ]
    
    spam_templates = [
        "URGENT: {subject} - {offer}! {call_to_action} now at {website}. {urgency}",
        "CONGRATULATIONS! You've been selected for {offer}. {call_to_action} - {urgency}",
        "{subject}: {offer} - {call_to_action} - {website} - Limited Time Only!",
        "MAKE MONEY FAST: {offer} - {urgency} - {call_to_action} - {website}",
        "FREE {subject} for first 100 customers! {offer} {call_to_action} {website}",
        "{subject} SALE! {offer} - {urgency} - {website}",
        "EXCLUSIVE OFFER: {subject} - {offer} - {call_to_action}",
        "IMPORTANT: Your {subject} subscription is about to expire. {offer} {call_to_action}",
        "Claim your FREE {subject} today! {offer} {website}",
        "Last chance! {offer} for {subject} - {call_to_action} {urgency}"
    ]
    
    subjects = [
        "membership", "product", "service", "subscription", "lottery", "account",
        "credit card", "loan", "investment", "insurance policy", "prescription",
        "vacation", "prize", "gift card", "bonus"
    ]
    
    offers = [
        "50% off", "buy one get one free", "exclusive discount", "limited time offer",
        "special promotion", "free trial", "no money down", "guaranteed results",
        "lowest rates ever", "no credit check", "zero interest", "instant approval"
    ]
    
    calls_to_action = [
        "Click here", "Sign up", "Register now", "Claim your prize", "Call today",
        "Visit our website", "Order now", "Apply online", "Subscribe today", 
        "Don't wait", "Respond immediately", "Take advantage"
    ]
    
    urgencies = [
        "Offer expires today", "Only 24 hours left", "Limited spots available",
        "While supplies last", "First come, first served", "Act now before it's too late",
        "Don't miss out", "One-time offer", "Today only", "Final notice"
    ]
    
    websites = [
        "www.special-offer.com", "www.amazing-deal.net", "www.claim-prize.com",
        "www.exclusive-savings.net", "www.best-rates-ever.com", "www.limited-time.net",
        "www.instant-approval.com", "www.guaranteed-results.net", "www.no-risk.com"
    ]
    
    # Generate spam emails
    spam_texts = []
    for _ in range(num_spam):
        template = np.random.choice(spam_templates)
        subject = np.random.choice(subjects)
        offer = np.random.choice(offers)
        call_to_action = np.random.choice(calls_to_action)
        urgency = np.random.choice(urgencies)
        website = np.random.choice(websites)
        
        spam_text = template.format(
            subject=subject,
            offer=offer,
            call_to_action=call_to_action,
            urgency=urgency,
            website=website
        )
        
        # Add some random spam phrases for more variety
        num_extra_phrases = np.random.randint(0, 4)
        extra_phrases = np.random.choice(spam_phrases, num_extra_phrases, replace=False)
        for phrase in extra_phrases:
            spam_text += f" {phrase}!"
        
        spam_texts.append(spam_text)
    
    # Create DataFrame
    texts = ham_texts + spam_texts
    labels = [0] * num_ham + [1] * num_spam
    
    # Shuffle the data
    indices = np.arange(len(texts))
    np.random.shuffle(indices)
    
    texts = [texts[i] for i in indices]
    labels = [labels[i] for i in indices]
    
    # Create DataFrame
    df = pd.DataFrame({
        'text': texts,
        'label': labels
    })
    
    # Save to CSV
    df.to_csv(args.output, index=False)
    print(f"Sample dataset saved to {args.output}")
    print(f"Total emails: {len(df)}")
    print(f"Ham emails: {len(df[df['label'] == 0])}")
    print(f"Spam emails: {len(df[df['label'] == 1])}")

if __name__ == '__main__':
    main()
