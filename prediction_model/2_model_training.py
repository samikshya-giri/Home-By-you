import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, f1_score, accuracy_score, auc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize
import os
import logging # Import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure the plots directory exists
os.makedirs('plots', exist_ok=True)

def plot_feature_importance(booster, feature_names, filename):
    """
    Plots the top N feature importances from an XGBoost model.
    Args:
        booster (xgb.Booster): The trained XGBoost booster model.
        feature_names (list): A list of actual feature names corresponding to the
                               order of features in the input data.
        filename (str): The path to save the plot.
    """
    fig, ax = plt.subplots(figsize=(14, 12))
    importance = booster.get_score(importance_type='gain')
    
    mapped_importance = {}
    for k, v in importance.items():
        try:
            # XGBoost `get_score` often returns 'f0', 'f1', etc. for feature names
            # Map these back to your actual feature names using the `feature_names` list
            # The 'f' prefix means feature. So 'f0' maps to feature_names[0].
            feature_idx = int(k[1:]) 
            mapped_importance[feature_names[feature_idx]] = v
        except (ValueError, IndexError):
            # Fallback for other types of keys or if index is out of bounds
            mapped_importance[k] = v

    importance_sorted = sorted(mapped_importance.items(), key=lambda x: x[1], reverse=True)[:20]
    features = [x[0] for x in importance_sorted]
    values = [x[1] for x in importance_sorted]
    
    sns.barplot(x=values, y=features, palette='viridis', ax=ax)
    plt.title('Top 20 Feature Importance (Gain)', fontsize=16)
    plt.xlabel('Gain', fontsize=14)
    plt.ylabel('Features', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Plot saved: {filename}")

def plot_confusion_matrix(y_true, y_pred, classes, filename, normalize=False):
    """
    Plots a confusion matrix.
    Args:
        y_true (np.array): True labels.
        y_pred (np.array): Predicted labels.
        classes (list): List of class names.
        filename (str): The path to save the plot.
        normalize (bool): Whether to normalize the confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                xticklabels=classes, yticklabels=classes, cmap='Blues',
                annot_kws={"size": 10}, cbar_kws={'shrink': 0.8})
    plt.title('Normalized Confusion Matrix' if normalize else 'Confusion Matrix', 
              fontsize=16, pad=20)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Plot saved: {filename}")

def plot_precision_recall(y_true, y_scores, class_names, filename):
    """
    Plots Precision-Recall curves for each class.
    Args:
        y_true (np.array): True labels.
        y_scores (np.array): Predicted probabilities for each class.
        class_names (list): List of class names.
        filename (str): The path to save the plot.
    """
    num_classes = len(class_names)
    # Ensure y_true_binarized has the correct number of classes
    y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))
    
    plt.figure(figsize=(14, 12))
    
    auc_scores = {}
    for i, class_name in enumerate(class_names):
        # Handle cases where a class might not be present in y_true_bin for precision_recall_curve
        # or where all scores are the same for a class.
        if np.sum(y_true_bin[:, i]) == 0 or len(np.unique(y_scores[:, i])) < 2:
            auc_scores[class_name] = np.nan # Mark as Not Available
            continue

        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_scores[:, i])
        # Check if precision and recall arrays are not empty before calculating AUC
        if len(precision) > 1 and len(recall) > 1:
            auc_score = auc(recall, precision) # Using sklearn.metrics.auc for more robustness
            auc_scores[class_name] = auc_score
        else:
            auc_scores[class_name] = np.nan

    # Sort classes by AUC score, handling NaN values
    sorted_classes = sorted(auc_scores.items(), key=lambda x: x[1] if not np.isnan(x[1]) else -1, reverse=True)
    
    for class_name, auc_score in sorted_classes:
        i = np.where(class_names == class_name)[0][0] # Get original index from the class_names array
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_scores[:, i])
        # Only plot if AUC was calculable
        if not np.isnan(auc_score):
            plt.plot(recall, precision, lw=2, 
                     label=f'{class_name} (AUC = {auc_score:.2f})')
        else:
            logging.warning(f"Skipping P-R curve for {class_name} due to insufficient data or constant predictions.")

    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('Precision-Recall Curve (Sorted by AUC)', fontsize=16)
    # Adjust legend position to avoid overlapping with plot if many classes
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, borderaxespad=0.) 
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Plot saved: {filename}")

def threshold_tuning_per_class(y_true, y_probs, class_names):
    """
    Performs threshold tuning for each class to maximize F1-score.
    Args:
        y_true (np.array): True labels.
        y_probs (np.array): Predicted probabilities for each class.
        class_names (list): List of class names.
    Returns:
        np.array: An array of optimal thresholds for each class.
    """
    best_thresholds = np.zeros(len(class_names))
    
    # Binarize true labels for per-class F1 score calculation
    y_true_bin = label_binarize(y_true, classes=np.arange(len(class_names)))
    
    logging.info("\n🔍 Threshold Tuning Results (Optimizing F1-score per class):")
    logging.info("{:<20} {:<15} {:<15} {:<15}".format(
        "Class", "Best Threshold", "F1 Score", "Support"))
    logging.info("-" * 65)
    
    for i in range(len(class_names)):
        best_f1 = 0
        best_thresh = 0.5 # Default threshold
        thresholds = np.linspace(0.01, 0.99, 100) # More granular thresholds
        support = np.sum(y_true == i) # Number of true instances for this class
        
        if support == 0:
            logging.info(f"{class_names[i]:<20} {'N/A':<15} {'N/A':<15} {support:<15}")
            # If no support, keep default threshold or 0.
            # Assign a very low threshold if you want to bias towards recall for rare classes,
            # but 0.5 is a reasonable default if not present in test set.
            best_thresholds[i] = 0.5 
            continue

        for thresh in thresholds:
            y_pred_thresh = (y_probs[:, i] >= thresh).astype(int)
            # Use `binary` average as we are evaluating one class against all others
            f1 = f1_score(y_true_bin[:, i], y_pred_thresh, average='binary', zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
                
        best_thresholds[i] = best_thresh
        logging.info("{:<20} {:<15.3f} {:<15.4f} {:<15}".format(
            class_names[i], best_thresh, best_f1, support))
    
    return best_thresholds

def main():
    output_dir = 'output'
    plots_dir = 'plots'
    
    os.makedirs(plots_dir, exist_ok=True)

    try:
        X_test = np.load(os.path.join(output_dir, 'X_test.npy'))
        y_test = np.load(os.path.join(output_dir, 'y_test.npy'))

        with open(os.path.join(output_dir, 'preprocessing.pkl'), 'rb') as f:
            prep = pickle.load(f)

        class_names = prep['category_label_encoder'].classes_
        # --- IMPORTANT: Get full feature names from preprocessing.pkl ---
        feature_names = prep['feature_names'] 
        
        logging.info("Successfully loaded model artifacts and test data.")

    except FileNotFoundError as e:
        logging.error(f"Error: Required files not found: {e}")
        logging.error("Please ensure the training script (1_data_preparation.py) has been run successfully and generated these files in the 'output/' directory.")
        return
    except Exception as e:
        logging.error(f"An error occurred during loading: {e}", exc_info=True)
        return

    booster = xgb.Booster()
    booster.load_model(os.path.join(output_dir, 'model.xgb'))
    
    # --- IMPORTANT: Pass feature_names to DMatrix ---
    dtest = xgb.DMatrix(X_test, enable_categorical=True, feature_names=feature_names) 
    
    # Predict probabilities (iteration_range for best_iteration if early stopping was used)
    y_probs = booster.predict(dtest, iteration_range=(0, booster.best_iteration))
    y_pred_initial = np.argmax(y_probs, axis=1) # Default 0.5 threshold for argmax

    f1_macro_initial = f1_score(y_test, y_pred_initial, average='macro')
    f1_weighted_initial = f1_score(y_test, y_pred_initial, average='weighted')
    accuracy_initial = accuracy_score(y_test, y_pred_initial)
    
    logging.info("\n📊 Initial Model Performance (Default Threshold):")
    logging.info(f"Accuracy: {accuracy_initial:.4f}")
    logging.info(f"F1 Macro: {f1_macro_initial:.4f}")
    logging.info(f"F1 Weighted: {f1_weighted_initial:.4f}")
    
    # Generate and print classification report
    report = classification_report(
        y_test, y_pred_initial, 
        target_names=class_names,
        digits=4,
        output_dict=True,
        zero_division=0 # Handle cases where a class has no true instances or no predicted instances
    )
    
    report_df = pd.DataFrame(report).transpose()
    logging.info("\n📈 Classification Report Summary (Default Threshold):")
    
    # Drop aggregation rows for display purposes, keep overall metrics
    if 'accuracy' in report_df.index:
        report_df_display = report_df.drop(['accuracy', 'macro avg', 'weighted avg'])
    else:
        report_df_display = report_df.drop(['macro avg', 'weighted avg'])

    # Sort by f1-score for better readability
    logging.info(report_df_display[['precision', 'recall', 'f1-score', 'support']]
                 .sort_values('f1-score', ascending=False)
                 .to_markdown(floatfmt=".3f"))
    
    report_output_path = os.path.join(plots_dir, 'classification_report_initial.md')
    with open(report_output_path, 'w') as f:
        f.write("# Initial Classification Report (Default Threshold)\n\n")
        f.write(report_df.to_markdown(floatfmt=".3f"))
    logging.info(f"Classification report saved: {report_output_path}")

    # Plotting functions
    plot_feature_importance(booster, feature_names, os.path.join(plots_dir, 'feature_importance.png'))
    plot_confusion_matrix(y_test, y_pred_initial, class_names, os.path.join(plots_dir, 'confusion_matrix.png'))
    plot_confusion_matrix(y_test, y_pred_initial, class_names, os.path.join(plots_dir, 'confusion_matrix_normalized.png'), normalize=True)
    plot_precision_recall(y_test, y_probs, class_names, os.path.join(plots_dir, 'precision_recall_curve.png'))
    
    # Perform threshold tuning
    best_thresholds = threshold_tuning_per_class(y_test, y_probs, class_names)
    
    # Apply tuned thresholds for final prediction
    y_pred_tuned = np.zeros_like(y_test)
    for i in range(len(y_test)):
        # Apply the per-class thresholding. The class with the highest probability
        # relative to its *optimal threshold* is chosen.
        # Add a small epsilon (1e-9) to thresholds to prevent division by zero if a threshold is exactly 0.
        adjusted_probs = y_probs[i] / (best_thresholds + 1e-9) 
        y_pred_tuned[i] = np.argmax(adjusted_probs)
    
    f1_macro_tuned = f1_score(y_test, y_pred_tuned, average='macro')
    f1_weighted_tuned = f1_score(y_test, y_pred_tuned, average='weighted')
    accuracy_tuned = accuracy_score(y_test, y_pred_tuned)

    logging.info("\n🎯 Final Performance After Threshold Tuning:")
    logging.info(f"Accuracy: {accuracy_tuned:.4f} (Improvement: {accuracy_tuned - accuracy_initial:+.4f})")
    logging.info(f"F1 Macro: {f1_macro_tuned:.4f} (Improvement: {f1_macro_tuned - f1_macro_initial:+.4f})")
    logging.info(f"F1 Weighted: {f1_weighted_tuned:.4f} (Improvement: {f1_weighted_tuned - f1_weighted_initial:+.4f})")
    
    # Save tuned predictions and thresholds
    np.save(os.path.join(output_dir, 'y_pred_tuned.npy'), y_pred_tuned)
    with open(os.path.join(output_dir, 'best_thresholds.pkl'), 'wb') as f:
        pickle.dump(best_thresholds, f)

    # Save misclassified indices (after tuning)
    misclassified_indices = np.where(y_test != y_pred_tuned)[0]
    misclassified_filename = os.path.join(output_dir, 'misclassified_indices_tuned.npy')
    np.save(misclassified_filename, misclassified_indices)
    
    logging.info("\n💾 Saved additional files in 'output' directory:")
    logging.info(f"- {os.path.join(output_dir, 'y_pred_tuned.npy')} (predictions after threshold tuning)")
    logging.info(f"- {os.path.join(output_dir, 'best_thresholds.pkl')} (optimal thresholds per class)")
    logging.info(f"- {misclassified_filename} (indices of misclassified samples after tuning)")
    logging.info(f"All plots saved in '{plots_dir}/' directory.")

if __name__ == "__main__":
    main()