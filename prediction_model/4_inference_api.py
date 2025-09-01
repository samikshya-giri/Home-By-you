from flask import Flask, request, jsonify
from flask_cors import CORS
import xgboost as xgb
import numpy as np
import pandas as pd
import pickle
import os
import logging

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

model = None
preprocessing_pipeline = None

# Exact 20 categories as specified
ALL_CATEGORIES = [
    'bathtub', 'bed', 'bench', 'bookshelf', 'bunk_bed', 'cabinet', 
    'chair', 'couch', 'desk', 'dining_table', 'lamp', 'mirror', 
    'office_chair', 'plant', 'rug', 'sink', 'sofa', 'table', 
    'tv_stand', 'wardrobe'
]

# Categories by type (updated to match the exact 20 categories)
seating_categories = ['chair', 'sofa', 'bench', 'couch', 'office_chair', 'bunk_bed']
table_categories = ['table', 'dining_table', 'desk', 'tv_stand']
storage_categories = ['cabinet', 'wardrobe', 'bookshelf']
functional_categories = ['bathtub', 'sink', 'bed']
decorative_categories = ['lamp', 'mirror', 'plant', 'rug']

# Enhanced default core items per room using only the 20 categories
default_core_items = {
    'livingroom': ['sofa', 'tv_stand', 'cabinet', 'chair', 'couch'],
    'bedroom': ['bed', 'wardrobe', 'chair', 'mirror'],
    'bathroom': ['bathtub', 'sink', 'cabinet', 'mirror'],
    'kitchen': ['cabinet', 'dining_table', 'sink', 'chair'],
    'diningroom': ['dining_table', 'chair', 'cabinet'],
    'office': ['desk', 'office_chair', 'bookshelf', 'cabinet'],
    'studyroom': ['desk', 'bookshelf', 'chair', 'lamp'],
    'balcony': ['plant', 'chair', 'table', 'bench'],
    'kidsroom': ['bunk_bed', 'desk', 'chair', 'bookshelf'],
    'guestroom': ['bed', 'wardrobe', 'chair', 'mirror'],
    'hallway': ['table', 'mirror', 'bench', 'cabinet'],
    'classroom': ['desk', 'chair', 'bookshelf', 'table']
}

# Room-specific item weights for exact core item prioritization
room_item_weights = {
    'livingroom': {'sofa': 4.0, 'tv_stand': 3.5, 'cabinet': 3.0},
    'bedroom': {'bed': 4.0, 'wardrobe': 3.5, 'chair': 3.0},
    'bathroom': {'bathtub': 4.0, 'sink': 3.5, 'cabinet': 3.0},
    'kitchen': {'cabinet': 4.0, 'dining_table': 3.5, 'sink': 3.0},
    'diningroom': {'dining_table': 4.0, 'chair': 3.5, 'cabinet': 3.0},
    'office': {'desk': 4.0, 'office_chair': 3.5, 'bookshelf': 3.0},
    'studyroom': {'desk': 4.0, 'bookshelf': 3.5, 'chair': 3.0},
    'balcony': {'plant': 4.0, 'chair': 3.5, 'table': 3.0},
    'kidsroom': {'bunk_bed': 4.0, 'desk': 3.5, 'chair': 3.0},
    'guestroom': {'bed': 4.0, 'wardrobe': 3.5, 'chair': 3.0},
    'hallway': {'table': 4.0, 'mirror': 3.5, 'bench': 3.0},
    'classroom': {'desk': 4.0, 'chair': 3.5, 'bookshelf': 3.0}
}


def load_assets():
    global model, preprocessing_pipeline
    output_dir = 'output'
    model_path = os.path.join(output_dir, 'model.xgb')
    preprocessing_path = os.path.join(output_dir, 'preprocessing.pkl')

    if not os.path.exists(model_path) or not os.path.exists(preprocessing_path):
        logging.error("Model or preprocessing pipeline not found.")
        exit("Required model files are missing.")

    try:
        model = xgb.Booster()
        model.load_model(model_path)
        logging.info(f"Model loaded from {model_path}")

        with open(preprocessing_path, 'rb') as f:
            preprocessing_pipeline = pickle.load(f)
        logging.info(f"Preprocessing pipeline loaded from {preprocessing_path}")

    except Exception as e:
        logging.error(f"Error loading model or preprocessing: {e}")
        exit("Failed to load model or preprocessing pipeline.")


@app.route('/')
def health_check():
    return jsonify({"message": "Prediction API is running"}), 200


def apply_domain_logic(room_type, is_seating, is_table, is_storage, predictions):
    """
    Enhanced domain logic to prioritize exact core items when no type is selected
    """
    prob_dict = {pred['category']: float(pred['probability']) for pred in predictions}
    adjusted = prob_dict.copy()
    
    # Get room-specific core items and weights
    room_core_items = default_core_items.get(room_type, [])
    room_weights = room_item_weights.get(room_type, {})
    
    # Room-based boosting using weighted approach
    for item in room_core_items:
        if item in adjusted:
            weight = room_weights.get(item, 2.0)
            adjusted[item] *= weight
    
    # Type-based boosting if user specifies
    if is_seating:
        for item in seating_categories:
            if item in adjusted:
                adjusted[item] *= 1.4
    
    if is_table:
        for item in table_categories:
            if item in adjusted:
                adjusted[item] *= 1.3
    
    if is_storage:
        for item in storage_categories:
            if item in adjusted:
                adjusted[item] *= 1.2
    
    # SPECIAL CASE: When no type is selected, STRONGLY prioritize the exact core items
    if not any([is_seating, is_table, is_storage]):
        # Get the exact top core items for this room from room_weights
        top_core_items = list(room_weights.keys())[:3]  # Get top 3 weighted items
        
        # Very strong boost for the exact core items
        for item in top_core_items:
            if item in adjusted:
                adjusted[item] *= 4.0  # Very strong boost
        
        # Moderate boost for other room-appropriate items
        for item in room_core_items:
            if item in adjusted and item not in top_core_items:
                adjusted[item] *= 2.0
        
        # REDUCE decorative items when no type specified
        for item in decorative_categories:
            if item in adjusted:
                adjusted[item] *= 0.3  # Reduce decorative items
    
    # Add variety through controlled randomness (less for core items)
    for item in adjusted:
        if item in room_core_items:
            # Less randomness for core items (2% variation)
            randomness = np.random.uniform(0.98, 1.02)
        else:
            # More randomness for non-core items (10% variation)
            randomness = np.random.uniform(0.9, 1.1)
        adjusted[item] *= randomness
    
    # Ensure minimum probability for all items
    min_prob = 0.001  # Very low minimum to prevent dominance
    for item in adjusted:
        adjusted[item] = max(adjusted[item], min_prob)
    
    # Normalize probabilities
    total = sum(adjusted.values())
    if total > 0:
        adjusted = {k: v / total for k, v in adjusted.items()}
    
    # Sort predictions
    sorted_preds = sorted(adjusted.items(), key=lambda x: x[1], reverse=True)
    
    # For no-type-selected case, ensure top 3 are from room_weights
    final_predictions = []
    if not any([is_seating, is_table, is_storage]):
        top_core_items = list(room_weights.keys())[:3]
        
        # First, try to include the exact top core items
        core_included = 0
        for core_item in top_core_items:
            if core_item in adjusted and core_included < 3:
                final_predictions.append((core_item, adjusted[core_item]))
                core_included += 1
        
        # If we don't have 3 core items, fill with next highest
        if len(final_predictions) < 3:
            for cat, prob in sorted_preds:
                if cat not in [p[0] for p in final_predictions] and len(final_predictions) < 3:
                    final_predictions.append((cat, prob))
    else:
        # Just take top 3 when type is specified
        final_predictions = sorted_preds[:3]
    
    return [{"category": cat, "probability": float(f"{prob:.4f}")} for cat, prob in final_predictions]


@app.route('/predict', methods=['POST'])
def predict():
    if model is None or preprocessing_pipeline is None:
        return jsonify({"error": "Model or preprocessing pipeline not loaded."}), 500

    try:
        data = request.get_json()
        logging.info(f"Prediction request: {data}")

        room_type = data.get('room_type', 'livingroom').lower()
        color = data.get('color', 'white').lower()
        material = data.get('material', 'wood').lower()
        scale_x = float(data.get('scale_x', 1.0))
        scale_y = float(data.get('scale_y', 1.0))
        scale_z = float(data.get('scale_z', 1.0))
        rotation_y = float(data.get('rotation_y', 0.0))
        x = float(data.get('x', 2.5))
        y = float(data.get('y', 0.0))
        z = float(data.get('z', 2.5))
        is_seating = int(data.get('is_seating', 0))
        is_table = int(data.get('is_table', 0))
        is_storage = int(data.get('is_storage', 0))

        temp_df = pd.DataFrame([{
            'room_type': room_type, 'color': color, 'material': material,
            'scale_x': scale_x, 'scale_y': scale_y, 'scale_z': scale_z,
            'rotation_y': rotation_y, 'x': x, 'y': y, 'z': z,
            'is_seating': is_seating, 'is_table': is_table, 'is_storage': is_storage
        }])

        enc = preprocessing_pipeline

        # Encode categorical features safely
        for col, encoder_name in [('room_type', 'room_type_encoder'), ('color', 'color_encoder'), ('material', 'material_encoder')]:
            encoder = enc.get(encoder_name, None)
            if encoder:
                val = data.get(col, '').lower()
                if val in encoder.classes_:
                    temp_df[f'{col}_enc'] = encoder.transform([val])[0]
                else:
                    temp_df[f'{col}_enc'] = np.random.choice(range(len(encoder.classes_)))
            else:
                temp_df[f'{col}_enc'] = 0

        # Feature engineering
        temp_df['volume'] = temp_df['scale_x'] * temp_df['scale_y'] * temp_df['scale_z']
        temp_df['aspect_ratio_xz'] = temp_df['scale_x'] / (temp_df['scale_z'] + 1e-6)
        temp_df['aspect_ratio_xy'] = temp_df['scale_x'] / (temp_df['scale_y'] + 1e-6)
        temp_df['distance_to_center'] = np.sqrt((temp_df['x'] - 2.5) ** 2 + (temp_df['z'] - 2.5) ** 2)
        temp_df['is_wall_near'] = ((temp_df['x'] < 1) | (temp_df['x'] > 4) | (temp_df['z'] < 1) | (temp_df['z'] > 4)).astype(int)
        temp_df['is_corner'] = (((temp_df['x'] < 1.5) & (temp_df['z'] < 1.5)) |
                                ((temp_df['x'] > 3.5) & (temp_df['z'] > 3.5)) |
                                ((temp_df['x'] < 1.5) & (temp_df['z'] > 3.5)) |
                                ((temp_df['x'] > 3.5) & (temp_df['z'] < 1.5))).astype(int)
        temp_df['wall_corner_interaction'] = temp_df['is_wall_near'] * temp_df['is_corner']

        # Scale numerical features
        for col in enc['numerical_features']:
            if col not in temp_df.columns:
                temp_df[col] = 0.0
        scaled_data = enc['scaler'].transform(temp_df[enc['numerical_features']])
        scaled_df = pd.DataFrame(scaled_data, columns=enc['numerical_features'])

        # Combine features
        final_df = scaled_df.copy()
        final_df['room_type_enc'] = temp_df['room_type_enc']
        final_df['color_enc'] = temp_df['color_enc']
        final_df['material_enc'] = temp_df['material_enc']
        final_df['is_seating'] = temp_df['is_seating']
        final_df['is_table'] = temp_df['is_table']
        final_df['is_storage'] = temp_df['is_storage']

        # Prepare XGBoost DMatrix
        X_predict = pd.DataFrame([{feat: final_df.get(feat, 0.0).iloc[0] for feat in enc['feature_names']}])
        dmatrix = xgb.DMatrix(X_predict.values, enable_categorical=True, feature_names=enc['feature_names'])

        # Predict
        probs = model.predict(dmatrix)[0]
        all_preds = [{"category": enc['category_label_encoder'].inverse_transform([i])[0], "probability": p} for i, p in enumerate(probs)]
        all_preds.sort(key=lambda x: x['probability'], reverse=True)

        top_preds = apply_domain_logic(room_type, is_seating, is_table, is_storage, all_preds)
        return jsonify({"predicted_category": top_preds[0]['category'], "top_predictions": top_preds})

    except Exception as e:
        logging.exception("Prediction error.")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    with app.app_context():
        load_assets()
    app.run(debug=True, host='0.0.0.0', port=5000)