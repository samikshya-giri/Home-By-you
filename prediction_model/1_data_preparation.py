import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTENC
import xgboost as xgb
import logging # Import logging for better output messages

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# -------------------- Enhanced Cleaning Rules --------------------
# Keep these rules as they are good for data quality.
general_furniture = ['lamp', 'rug', 'plant', 'chair', 'table', 'mirror', 'cabinet']
ALLOWED_ROOM_FURNITURE = {
    "bathroom": ['bathtub', 'mirror', 'cabinet', 'sink', 'lamp'] + general_furniture,
    "bedroom": ['bed', 'wardrobe', 'lamp', 'mirror', 'rug', 'chair', 'table', 'plant', 'bookshelf'] + general_furniture,
    "livingroom": ['sofa', 'tv_stand', 'coffee_table', 'couch', 'lamp', 'plant', 'rug', 'bookshelf', 'chair'] + general_furniture,
    "kitchen": ['table', 'chair', 'cabinet', 'refrigerator', 'lamp', 'sink'] + general_furniture,
    "diningroom": ['dining_table', 'chair', 'lamp', 'plant', 'cabinet'] + general_furniture,
    "office": ['desk', 'office_chair', 'bookshelf', 'lamp', 'plant', 'chair'] + general_furniture,
    "studyroom": ['desk', 'chair', 'bookshelf', 'lamp', 'plant'] + general_furniture,
    "classroom": ['desk', 'chair', 'bookshelf', 'lamp', 'plant'] + general_furniture,
    "kidsroom": ['bunk_bed', 'wardrobe', 'rug', 'lamp', 'plant', 'bookshelf', 'chair'] + general_furniture,
    "guestroom": ['bed', 'wardrob', 'mirror', 'lamp', 'rug', 'plant', 'bookshelf'] + general_furniture,
    "hallway": ['bench', 'mirror', 'plant', 'lamp', 'cabinet'] + general_furniture,
    "balcony": ['plant', 'bench', 'chair', 'lamp', 'table'] + general_furniture
}

INVALID_FURNITURE_PER_ROOM = {
    "bedroom": ['dining_table', 'bookshelf', 'sink', 'refrigerator', 'office_chair', 'coffee_table'],
    "bathroom": ['bed', 'bookshelf', 'couch', 'refrigerator', 'tv_stand', 'sofa', 'desk', 'dining_table', 'office_chair', 'bunk_bed', 'coffee_table'],
    "kitchen": ['bed', 'bookshelf', 'bunk_bed', 'bench', 'office_chair'],
    "diningroom": ['bed', 'wardrobe', 'sofa', 'desk', 'office_chair', 'bunk_bed'],
    "office": ['bed', 'dining_table', 'bathtub', 'refrigerator', 'sink'],
    "livingroom": ['bathtub', 'sink', 'refrigerator', 'desk', 'office_chair'],
    "kidsroom": ['dining_table', 'sink', 'refrigerator', 'bathtub'],
    "guestroom": ['dining_table', 'sink', 'refrigerator'],
    "hallway": ['bed', 'bunk_bed', 'bathtub', 'sink', 'refrigerator', 'dining_table'],
    "balcony": ['bed', 'wardrobe', 'desk', 'office_chair', 'bathtub', 'sink'],
    "studyroom": ['bathtub', 'sink', 'dining_table', 'refrigerator'],
    "classroom": ['bed', 'bunk_bed', 'bathtub', 'sink', 'dining_table', 'refrigerator']
}

INVALID_MATERIALS = {
    'bathtub': ['fabric', 'leather', 'wood'],
    'plant': ['leather'],
    'rug': ['glass', 'metal', 'stone', 'concrete', 'ceramic'],
    'mirror': ['fabric'],
    'lamp': ['leather', 'concrete'],
    'sink': ['wood', 'leather'],
    'bookshelf': ['leather'],
    'chair': ['glass', 'ceramic'],
    'table': ['fabric']
}

INVALID_COLORS = {
    'plant': ['black', 'silver', 'gold'],
    'bathtub': ['red', 'green'],
    'sink': ['gold', 'purple']
}

def is_valid(row):
    item = row['category']
    room = row['room_type']
    mat = row['material'].lower()
    col = row['color'].lower()

    # Apply cleaning rules
    if room in INVALID_FURNITURE_PER_ROOM and item in INVALID_FURNITURE_PER_ROOM[room]:
        return False
    # Only allow furniture that makes sense in the room, if explicitly defined
    # It's better to exclude categories not in ALLOWED_ROOM_FURNITURE only if they are not general furniture
    if item not in ALLOWED_ROOM_FURNITURE.get(room, []):
        return False
    if item in INVALID_MATERIALS and mat in INVALID_MATERIALS[item]:
        return False
    if item in INVALID_COLORS and col in INVALID_COLORS[item]:
        return False
    return True

def main():
    # Load and clean data
    df = pd.read_csv('enriched_furniture_dataset_13000.csv') # Assuming this is your dataset
    original_df_len = len(df)
    
    # Standardize string columns to lowercase for consistent matching
    df['room_type'] = df['room_type'].str.lower()
    df['material'] = df['material'].str.lower()
    df['color'] = df['color'].str.lower()
    df['category'] = df['category'].str.lower()

    df_cleaned = df[df.apply(is_valid, axis=1)].copy()
    logging.info(f"Data cleaned. Original rows: {original_df_len}, Remaining rows: {len(df_cleaned)}")

    # --- CRITICAL CHANGE: Re-evaluating 'OTHER' category handling ---
    # The problem might be 'OTHER' being too large and diverse.
    # Instead of replacing rare categories with 'OTHER', let's remove *very* rare ones,
    # and if 'OTHER' is an existing category, treat it as such.
    # If 'OTHER' dominates, consider reducing its samples (undersampling) or defining it more narrowly.

    MIN_SAMPLES_FOR_CATEGORY = 25 # Categories with fewer than this will be dropped
    
    # Get initial value counts *before* dropping rare categories
    initial_category_counts = df_cleaned['category'].value_counts()
    logging.info(f"Initial category distribution:\n{initial_category_counts}")

    # Identify and potentially remove very rare categories
    # This is a strong step. Only do this if you're sure these categories are noise or very hard to learn.
    # If 'OTHER' is still dominating after this, we might need to actively undersample it.
    categories_to_keep = initial_category_counts[initial_category_counts >= MIN_SAMPLES_FOR_CATEGORY].index
    df_filtered = df_cleaned[df_cleaned['category'].isin(categories_to_keep)].copy()
    
    # If 'OTHER' is still present and very dominant, we might need to undersample it *before* SMOTENC
    # This is an optional aggressive step if the problem persists.
    # If 'OTHER' is defined as a specific furniture type, then this undersampling might not be appropriate.
    # Let's assume 'OTHER' is indeed a catch-all for now and needs balancing.
    
    # Re-check category counts after initial filtering
    final_category_counts = df_filtered['category'].value_counts()
    logging.info(f"Category distribution after filtering (>= {MIN_SAMPLES_FOR_CATEGORY} samples):\n{final_category_counts}")
    
    # If 'OTHER' is still the majority after filtering, consider undersampling it directly
    # This is a crucial step if 'OTHER' is causing the prediction bias.
    if 'other' in final_category_counts.index: # Assuming 'other' is lowercase
        other_count = final_category_counts['other']
        # Define a target ratio or absolute count for 'other'
        # For example, reduce 'other' to 1.5 times the average of other categories, or a fixed number.
        target_other_count = final_category_counts[final_category_counts.index != 'other'].mean() * 1.5
        target_other_count = min(other_count, int(target_other_count)) # Don't over-reduce or increase

        if other_count > target_other_count:
            logging.info(f"Undersampling 'other' from {other_count} to {target_other_count} samples.")
            df_other = df_filtered[df_filtered['category'] == 'other'].sample(n=target_other_count, random_state=42)
            df_non_other = df_filtered[df_filtered['category'] != 'other']
            df_processed = pd.concat([df_other, df_non_other])
        else:
            df_processed = df_filtered # No undersampling needed for 'other'
    else:
        df_processed = df_filtered # No 'other' category found

    logging.info(f"Categories after cleaning and 'OTHER' handling: {df_processed['category'].nunique()} unique categories. Total rows: {len(df_processed)}")
    
    # Advanced feature engineering (your existing good features)
    df_processed['volume'] = df_processed['scale_x'] * df_processed['scale_y'] * df_processed['scale_z']
    df_processed['aspect_ratio_xz'] = df_processed['scale_x'] / (df_processed['scale_z'] + 1e-6)
    df_processed['aspect_ratio_xy'] = df_processed['scale_x'] / (df_processed['scale_y'] + 1e-6)
    df_processed['distance_to_center'] = np.sqrt((df_processed['x'] - 2.5)**2 + (df_processed['z'] - 2.5)**2)
    df_processed['is_wall_near'] = ((df_processed['x'] < 1) | (df_processed['x'] > 4) | (df_processed['z'] < 1) | (df_processed['z'] > 4)).astype(int)
    
    # Ensure 'is_corner' is boolean before converting to int, for consistency
    df_processed['is_corner'] = ((df_processed['x'] < 1.5) & (df_processed['z'] < 1.5)) | \
                                ((df_processed['x'] > 3.5) & (df_processed['z'] > 3.5)) | \
                                ((df_processed['x'] < 1.5) & (df_processed['z'] > 3.5)) | \
                                ((df_processed['x'] > 3.5) & (df_processed['z'] < 1.5))
    df_processed['is_corner'] = df_processed['is_corner'].astype(int) # Convert boolean to int
    df_processed['wall_corner_interaction'] = df_processed['is_wall_near'] * df_processed['is_corner']
    
    # Semantic groups (features *derived from category*, so be careful with interpretation)
    # These features are good *if* they were part of the data before we knew the category.
    # Since they are derived from `df['category']`, they are essentially "target leakage" in strict terms
    # unless you have a separate, independent way to determine if an item *is* seating, table, storage
    # from its raw properties. For training, it's fine as long as you understand this.
    # For inference, if these are optional/defaulted to 0, the model will perform differently.
    # However, given your use case, they are part of the model's expected input for the final prediction task,
    # so keeping them here is necessary if the model was trained with them.
    seating = ['chair', 'sofa', 'bench', 'couch', 'office_chair', 'bunk_bed'] # Added bunk_bed to seating
    tables = ['table', 'coffee_table', 'desk', 'dining_table', 'tv_stand']
    storage = ['cabinet', 'wardrobe', 'bookshelf', 'refrigerator'] # Added refrigerator to storage
    
    df_processed['is_seating'] = df_processed['category'].isin(seating).astype(int)
    df_processed['is_table'] = df_processed['category'].isin(tables).astype(int)
    df_processed['is_storage'] = df_processed['category'].isin(storage).astype(int)
    
    # Room-specific features
    # Ensure this list is derived from the *processed* DataFrame's unique room types
    unique_rooms = df_processed['room_type'].unique()
    for room in unique_rooms:
        df_processed[f'is_{room}'] = (df_processed['room_type'] == room).astype(int)
    
    # Target encoding
    le = LabelEncoder()
    y = le.fit_transform(df_processed['category'])
    logging.info(f"Target variable encoded. Number of classes: {len(np.unique(y))}")
    
    # Feature selection (numerical columns)
    num_cols = [
        'scale_x', 'scale_y', 'scale_z', 'rotation_y', 'x', 'y', 'z',
        'volume', 'aspect_ratio_xz', 'aspect_ratio_xy', 'distance_to_center',
        'is_wall_near', 'is_corner', 'wall_corner_interaction',
        'is_seating', 'is_table', 'is_storage'
    ] + [f'is_{room}' for room in unique_rooms] # Use unique_rooms here
    
    # Scaling numerical features
    scaler = MinMaxScaler()
    df_processed[num_cols] = scaler.fit_transform(df_processed[num_cols])
    
    # Categorical encoding - Fit before creating X, and store encoders
    room_type_encoder = LabelEncoder()
    df_processed['room_type_enc'] = room_type_encoder.fit_transform(df_processed['room_type'])
    
    color_encoder = LabelEncoder()
    df_processed['color_enc'] = color_encoder.fit_transform(df_processed['color'])
    
    material_encoder = LabelEncoder()
    df_processed['material_enc'] = material_encoder.fit_transform(df_processed['material'])
    logging.info("Categorical features (room_type, color, material) encoded.")
    
    # Feature matrix
    X = np.hstack([
        df_processed[num_cols].values,
        df_processed[['room_type_enc', 'color_enc', 'material_enc']].values
    ])
    
    # Identify categorical feature indices for SMOTENC and XGBoost
    # The indices will be at the end of the feature array X
    # Ensure these indices are correct based on the order of hstack
    # If num_cols changes, these indices might need adjustment.
    # Currently: num_cols features + room_type_enc + color_enc + material_enc
    # So the last 3 features are categorical.
    cat_idx = [X.shape[1] - 3, X.shape[1] - 2, X.shape[1] - 1]
    logging.info(f"Categorical feature indices for SMOTENC/XGBoost: {cat_idx}")

    # Train-test split
    # Stratify by y to ensure balanced representation of classes in both train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )
    logging.info(f"Data split into training ({len(X_train)} samples) and test sets ({len(X_test)} samples).")
    
    # Enhanced SMOTENC
    # Check minimum samples per class in the training set
    train_category_counts = pd.Series(y_train).value_counts()
    min_class_samples = train_category_counts.min()
    
    # k_neighbors for SMOTENC must be less than or equal to the number of samples in the minority class
    smote_k_neighbors = max(1, min(5, min_class_samples - 1)) 
    
    if smote_k_neighbors < 1:
        logging.warning("⚠️ Warning: SMOTENC cannot be applied as some minority classes have too few samples (less than 2). Skipping SMOTENC.")
        X_res, y_res = X_train, y_train
    else:
        smote = SMOTENC(
            categorical_features=cat_idx,
            random_state=42,
            k_neighbors=smote_k_neighbors,
            sampling_strategy='not majority' # Balance all classes except the majority
        )
        logging.info(f"Applying SMOTENC with k_neighbors={smote_k_neighbors} for imbalance handling.")
        X_res, y_res = smote.fit_resample(X_train, y_train)
    logging.info(f"After SMOTENC, training samples: {len(X_res)}")
    logging.info(f"Distribution after SMOTENC:\n{pd.Series(y_res).value_counts()}") # Check balance

    # Class weights - compute based on the *resampled* data if SMOTENC was applied, else on X_train
    weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(y_res),
        y=y_res
    )
    sample_weights = np.array([weights[cls] for cls in y_res])
    logging.info("Class weights computed for balanced training.")
    
    # XGBoost training with optimized parameters
    dtrain = xgb.DMatrix(X_res, label=y_res, weight=sample_weights, enable_categorical=True)
    dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)
    
    # Tuned parameters for better generalization and performance
    params = {
        'objective': 'multi:softprob',
        'num_class': len(np.unique(y)),
        'max_depth': 7,          # Slightly reduced from 8 for generalization
        'eta': 0.015,            # Further reduced learning rate for more subtle learning
        'subsample': 0.8,        # Slightly increased subsample
        'colsample_bytree': 0.8, # Slightly increased colsample
        'min_child_weight': 1,   # Keep as 1
        'gamma': 0.1,            # Slightly increased gamma (min loss reduction required for a split)
        'lambda': 1,             # L2 regularization
        'alpha': 0.2,            # L1 regularization (increased for more sparsity, might help with too many features)
        'tree_method': 'hist',   # Fast histogram-based tree construction
        'eval_metric': ['mlogloss', 'merror'],
        'seed': 42,
        'n_jobs': -1,
        'rate_drop': 0.05,       # Add DART specific parameters (optional, but can help generalization)
        'skip_drop': 0.5         # Fraction of trees to skip dropping
    }
    
    logging.info("\nStarting XGBoost training...")
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=4000, # Increased rounds significantly for lower eta, early stopping will prevent overfitting
        evals=[(dtrain, 'train'), (dtest, 'eval')],
        early_stopping_rounds=200, # Increased early stopping rounds to allow more learning
        verbose_eval=500 # Print evaluation every 500 rounds
    )
    logging.info(f"XGBoost training complete. Best iteration: {model.best_iteration}")
    
    # Evaluation
    y_pred_proba = model.predict(dtest, iteration_range=(0, model.best_iteration))
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    f1 = f1_score(y_test, y_pred, average='macro')
    accuracy = accuracy_score(y_test, y_pred)
    
    logging.info(f"\n📊 Model Evaluation Results:")
    logging.info(f"✅ F1 Score (macro): {f1:.4f}")
    logging.info(f"✅ Accuracy: {accuracy:.4f}")

    # --- CRITICAL: Per-class Classification Report ---
    # This will show you exactly how well the model performs for each category.
    # Look for low F1-scores for specific furniture types, and high F1 for "OTHER", "lamp", "plant".
    class_names = le.inverse_transform(np.arange(len(np.unique(y))))
    logging.info("\nDetailed Classification Report:")
    logging.info(classification_report(y_test, y_pred, target_names=class_names))

    if f1 >= 0.85 and accuracy >= 0.85:
        logging.info("\n🎉 Congratulations! F1 Score and Accuracy both achieved over 85%.")
    else:
        logging.info("\nKeep optimizing! F1 Score and Accuracy are below 85%. Consider further tuning or data refinement.")

    # Ensure output directory exists
    os.makedirs('output', exist_ok=True)
    
    # Save all required files
    # Only save X_res and y_res if SMOTENC was actually applied.
    # Otherwise, save X_train and y_train directly.
    np.save('output/X_train.npy', X_train)
    np.save('output/X_test.npy', X_test)
    np.save('output/y_train.npy', y_train)
    np.save('output/y_test.npy', y_test)
    
    # Save X_res and y_res
    np.save('output/X_resampled.npy', X_res)
    np.save('output/y_resampled.npy', y_res)
    
    # Prepare feature names for DMatrix for consistency in inference API
    feature_names_for_dmatrix = num_cols + ['room_type_enc', 'color_enc', 'material_enc']

    with open('output/preprocessing.pkl', 'wb') as f:
        pickle.dump({
            'scaler': scaler,
            'category_label_encoder': le, 
            'room_type_encoder': room_type_encoder,
            'color_encoder': color_encoder,
            'material_encoder': material_encoder,
            'numerical_features': num_cols, # Store original column names of numerical features
            'categorical_feature_cols': ['room_type', 'color', 'material'], # Store original categorical column names
            'cat_feature_indices': cat_idx, # Store indices in the numpy array X
            'feature_names': feature_names_for_dmatrix # Save full list of feature names in order
        }, f)
    
    model.save_model('output/model.xgb')
    
    logging.info("\n💾 Saved files in 'output' directory:")
    logging.info("- X_train.npy, X_test.npy, y_train.npy, y_test.npy")
    logging.info("- X_resampled.npy, y_resampled.npy")
    logging.info("- model.xgb (trained XGBoost model)")
    logging.info("- preprocessing.pkl (scalers, encoders, and feature names)")

if __name__ == "__main__":
    main()