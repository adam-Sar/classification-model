import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import numpy as np
from classifier import classify_article
import json
from collections import defaultdict

def load_test_data(file_path):
    """Load test data from a JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def evaluate_model(test_data):
    """
    Evaluate the model's performance on test data
    
    Args:
        test_data: List of dictionaries containing 'text' and 'label' keys
    """
    predictions = []
    true_labels = []
    
    # Track metrics for threatening articles only
    threatening_metrics = {
        'total': 0,
        'correct': 0,
        'category_correct': 0
    }
    
    # Track metrics for non-threatening articles only
    non_threatening_metrics = {
        'total': 0,
        'correct': 0
    }
    
    # Track category-wise metrics
    category_metrics = defaultdict(lambda: {'total': 0, 'correct': 0})
    
    for item in test_data:
        # Get model prediction
        result = classify_article(item['text'])
        predicted_threat = result['threatening']
        true_threat = item['label']
        
        # Store predictions and true labels
        predictions.append(predicted_threat)
        true_labels.append(true_threat)
        
        # Update metrics based on whether it's threatening or not
        if true_threat:
            threatening_metrics['total'] += 1
            if predicted_threat:
                threatening_metrics['correct'] += 1
                
                # Handle category prediction
                predicted_category = result['category']
                true_category = item['category']
                
                # Check if the true category is in the predicted categories
                if isinstance(predicted_category, list):
                    # If model predicts multiple categories, consider it correct if true category is in the list
                    if true_category in predicted_category:
                        threatening_metrics['category_correct'] += 1
                        category_metrics[true_category]['correct'] += 1
                else:
                    if predicted_category == true_category:
                        threatening_metrics['category_correct'] += 1
                        category_metrics[true_category]['correct'] += 1
                
                category_metrics[true_category]['total'] += 1
        else:
            non_threatening_metrics['total'] += 1
            if not predicted_threat:
                non_threatening_metrics['correct'] += 1
    
    # Calculate overall metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    conf_matrix = confusion_matrix(true_labels, predictions)
    
    # Print detailed results
    print("\n=== Overall Model Evaluation Results ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    print("\n=== Detailed Classification Report ===")
    print(classification_report(true_labels, predictions, target_names=['Non-threatening', 'Threatening']))
    
    print("\n=== Confusion Matrix ===")
    print("Predicted →")
    print("Actual ↓")
    print("          Non-threatening  Threatening")
    print(f"Non-threatening    {conf_matrix[0][0]:^12}    {conf_matrix[0][1]:^12}")
    print(f"Threatening        {conf_matrix[1][0]:^12}    {conf_matrix[1][1]:^12}")
    
    # Print threatening vs non-threatening specific metrics
    if threatening_metrics['total'] > 0:
        threatening_accuracy = threatening_metrics['correct'] / threatening_metrics['total']
        category_accuracy = threatening_metrics['category_correct'] / threatening_metrics['total']
        
        print("\n=== Threatening Articles Metrics ===")
        print(f"Accuracy: {threatening_accuracy:.4f}")
        print(f"Category Classification Accuracy: {category_accuracy:.4f}")
        
        print("\n=== Category-wise Performance ===")
        for category, metrics in category_metrics.items():
            if metrics['total'] > 0:
                cat_accuracy = metrics['correct'] / metrics['total']
                print(f"{category}: {cat_accuracy:.4f} ({metrics['correct']}/{metrics['total']})")
    
    if non_threatening_metrics['total'] > 0:
        non_threatening_accuracy = non_threatening_metrics['correct'] / non_threatening_metrics['total']
        print("\n=== Non-threatening Articles Metrics ===")
        print(f"Accuracy: {non_threatening_accuracy:.4f}")

if __name__ == "__main__":
    # Load test data
    test_data = load_test_data("test_data.json")
    
    # Evaluate model
    evaluate_model(test_data) 