from flask import Flask, request, jsonify
from flask_cors import CORS
import xgboost as xgb
import numpy as np
import pandas as pd
import pickle
import os
import logging
from sklearn.preprocessing import LabelEncoder # Assuming LabelEncoder is used

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

model = None
preprocessing_pipeline = None

seating_categories = ['chair', 'sofa', 'bench', 'couch', 'office_chair', 'bunk_bed']
table_categories = ['table', 'coffee_table', 'desk', 'dining_table', 'tv_stand']
storage_categories = ['cabinet', 'wardrobe', 'bookshelf', 'refrigerator']

def load_assets():
    global model, preprocessing_pipeline

    output_dir = 'output'
    model_path = os.path.join(output_dir, 'model.xgb')
    preprocessing_path = os.path.join(output_dir, 'preprocessing.pkl')

    if not os.path.exists(model_path) or not os.path.exists(preprocessing_path):
        logging.error("Model or preprocessing pipeline not found. Please ensure 'model.xgb' and 'preprocessing.pkl' are in the 'output' directory.")
        # Do not exit here in production, perhaps serve an error page or a "down for maintenance" message.
        # For development, exiting is fine.
        exit("Required model files are missing.")

    try:
        model = xgb.Booster()
        model.load_model(model_path)
        logging.info(f"Model loaded from {model_path}")

        with open(preprocessing_path, 'rb') as f:
            preprocessing_pipeline = pickle.load(f)
        logging.info(f"Preprocessing pipeline loaded from {preprocessing_path}")

        # Post-load check for encoders and their classes
        if 'color_encoder' in preprocessing_pipeline and hasattr(preprocessing_pipeline['color_encoder'], 'classes_'):
            logging.info(f"Color encoder classes: {preprocessing_pipeline['color_encoder'].classes_.tolist()}")
        if 'room_type_encoder' in preprocessing_pipeline and hasattr(preprocessing_pipeline['room_type_encoder'], 'classes_'):
            logging.info(f"Room type encoder classes: {preprocessing_pipeline['room_type_encoder'].classes_.tolist()}")
        if 'material_encoder' in preprocessing_pipeline and hasattr(preprocessing_pipeline['material_encoder'], 'classes_'):
            logging.info(f"Material encoder classes: {preprocessing_pipeline['material_encoder'].classes_.tolist()}")

    except Exception as e:
        logging.error(f"Error loading model or preprocessing: {e}")
        exit("Failed to load model or preprocessing pipeline. Check file integrity and permissions.")

@app.route('/')
def health_check():
    return jsonify({"message": "Prediction API is running"}), 200

def apply_domain_logic(room_type, is_seating, is_table, is_storage, predictions, encoder, num_top=3):
    prob_dict = {pred['category']: float(pred['probability']) for pred in predictions}
    adjusted = dict(prob_dict)

    def boost(items, factor):
        for item in items:
            if item in adjusted:
                adjusted[item] *= factor

    def penalize(items, factor):
        for item in items:
            if item in adjusted:
                adjusted[item] *= factor

    # Room-based logic
    room_rules = {
        'bathroom': (['sink', 'bathtub', 'toilet', 'mirror', 'shower'], ['bookshelf', 'bed', 'sofa']),
        'livingroom': (['couch', 'sofa', 'tv_stand', 'coffee_table'], ['toilet', 'sink']),
        'bedroom': (['bed', 'wardrobe', 'nightstand'], ['toilet', 'sink', 'tv_stand']),
        'kitchen': (['refrigerator', 'cabinet', 'dining_table'], ['bed', 'bookshelf']),
        'diningroom': (['dining_table', 'chair', 'cabinet'], ['bed', 'toilet']),
        'office': (['desk', 'office_chair', 'bookshelf'], ['bed', 'bathtub']),
        'studyroom': (['desk', 'office_chair', 'bookshelf'], ['bed', 'bathtub']),
        'balcony': ([], ['bed', 'toilet']), # Add rules for balcony if applicable
        'classroom': (['desk', 'chair', 'whiteboard'], ['bed', 'toilet']), # Add rules for classroom
        'guestroom': (['bed', 'wardrobe'], ['toilet', 'sink']), # Add rules for guestroom
        'hallway': (['console_table', 'mirror', 'coat_rack'], ['bed', 'dining_table']), # Add rules for hallway
        'kidsroom': (['bunk_bed', 'toy_chest', 'desk'], ['toilet', 'sink']) # Add rules for kidsroom
    }

    if room_type in room_rules:
        boost(room_rules[room_type][0], 1.5)
        penalize(room_rules[room_type][1], 0.1)

    # Boolean property logic
    if is_seating:
        boost(seating_categories, 2.0)
        # Penalize non-seating items, but be careful not to zero out everything
        for item in adjusted:
            if item not in seating_categories:
                adjusted[item] *= 0.5
    if is_table:
        boost(table_categories, 2.0)
        for item in adjusted:
            if item not in table_categories:
                adjusted[item] *= 0.5
    if is_storage:
        boost(storage_categories, 2.0)
        for item in adjusted:
            if item not in storage_categories:
                adjusted[item] *= 0.5

    # Re-normalize probabilities if you applied significant boosts/penalties
    # total_prob = sum(adjusted.values())
    # if total_prob > 0:
    #     for key in adjusted:
    #         adjusted[key] /= total_prob

    sorted_preds = sorted(adjusted.items(), key=lambda x: x[1], reverse=True)
    return [{"category": cat, "probability": f"{prob:.4f}"} for cat, prob in sorted_preds[:num_top]]


@app.route('/predict', methods=['POST'])
def predict():
    if model is None or preprocessing_pipeline is None:
        return jsonify({"error": "Model or preprocessing pipeline not loaded. Please check server logs."}), 500

    try:
        data = request.get_json()
        logging.info(f"Prediction request: {data}")

        # Extract input with robust defaults using .get()
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

        # Create a temporary DataFrame for the single prediction
        temp_df = pd.DataFrame([{
            'room_type': room_type, 'color': color, 'material': material,
            'scale_x': scale_x, 'scale_y': scale_y, 'scale_z': scale_z,
            'rotation_y': rotation_y, 'x': x, 'y': y, 'z': z,
            'is_seating': is_seating, 'is_table': is_table, 'is_storage': is_storage
        }])

        # Feature Engineering (must match what was done during training)
        temp_df['volume'] = temp_df['scale_x'] * temp_df['scale_y'] * temp_df['scale_z']
        temp_df['aspect_ratio_xz'] = temp_df['scale_x'] / (temp_df['scale_z'] + 1e-6)
        temp_df['aspect_ratio_xy'] = temp_df['scale_x'] / (temp_df['scale_y'] + 1e-6)
        temp_df['distance_to_center'] = np.sqrt((temp_df['x'] - 2.5) ** 2 + (temp_df['z'] - 2.5) ** 2)
        temp_df['is_wall_near'] = ((temp_df['x'] < 1) | (temp_df['x'] > 4) | (temp_df['z'] < 1) | (temp_df['z'] > 4)).astype(int)
        temp_df['is_corner'] = (((temp_df['x'] < 1.5) & (temp_df['z'] < 1.5)) | ((temp_df['x'] > 3.5) & (temp_df['z'] > 3.5)) |
                                 ((temp_df['x'] < 1.5) & (temp_df['z'] > 3.5)) | ((temp_df['x'] > 3.5) & (temp_df['z'] < 1.5))).astype(int)
        temp_df['wall_corner_interaction'] = temp_df['is_wall_near'] * temp_df['is_corner']

        enc = preprocessing_pipeline

        # --- Handle categorical encoding with unseen labels ---
        # Get the default values (e.g., the most frequent one from training)
        # This assumes your encoders are LabelEncoders and have a 'classes_' attribute.
        # You might need to adjust the 'default_value' if your training set is very specific.
        default_room_type_enc = 0 # Or a common room type's encoded value
        if 'room_type_encoder' in enc and len(enc['room_type_encoder'].classes_) > 0:
            default_room_type_enc = enc['room_type_encoder'].transform([enc['room_type_encoder'].classes_[0]])[0] # Use first class as default
        
        default_color_enc = 0 # Or a common color's encoded value
        if 'color_encoder' in enc and len(enc['color_encoder'].classes_) > 0:
            default_color_enc = enc['color_encoder'].transform([enc['color_encoder'].classes_[0]])[0] # Use first class as default

        default_material_enc = 0 # Or a common material's encoded value
        if 'material_encoder' in enc and len(enc['material_encoder'].classes_) > 0:
            default_material_enc = enc['material_encoder'].transform([enc['material_encoder'].classes_[0]])[0] # Use first class as default

        try:
            temp_df['room_type_enc'] = enc['room_type_encoder'].transform([room_type])[0]
        except ValueError:
            logging.warning(f"Unseen room_type '{room_type}', using default encoded value: {default_room_type_enc}")
            temp_df['room_type_enc'] = default_room_type_enc

        try:
            temp_df['color_enc'] = enc['color_encoder'].transform([color])[0]
        except ValueError:
            logging.warning(f"Unseen color '{color}', using default encoded value: {default_color_enc}")
            temp_df['color_enc'] = default_color_enc

        try:
            temp_df['material_enc'] = enc['material_encoder'].transform([material])[0]
        except ValueError:
            logging.warning(f"Unseen material '{material}', using default encoded value: {default_material_enc}")
            temp_df['material_enc'] = default_material_enc
        # --- End of categorical encoding handling ---

        # Scale numerical features
        numerical_features_to_scale = enc['numerical_features']
        
        # Ensure all expected numerical features are present in temp_df before scaling
        # and fill any missing with 0.0 (or a more appropriate default like mean/median)
        for col in numerical_features_to_scale:
            if col not in temp_df.columns:
                temp_df[col] = 0.0 # This might need to be reconsidered. If a feature is truly missing, 0.0 might not be the best imputation.

        scaled_data = enc['scaler'].transform(temp_df[numerical_features_to_scale])
        scaled = pd.DataFrame(scaled_data, columns=numerical_features_to_scale)

        # Combine scaled numerical features with encoded categorical features
        final_df = scaled.copy()
        final_df['room_type_enc'] = temp_df['room_type_enc']
        final_df['color_enc'] = temp_df['color_enc']
        final_df['material_enc'] = temp_df['material_enc']
        final_df['is_seating'] = temp_df['is_seating'] # Boolean features are already numeric (0 or 1)
        final_df['is_table'] = temp_df['is_table']
        final_df['is_storage'] = temp_df['is_storage']

        # If your model's feature names include one-hot encoded room types (e.g., 'is_bedroom')
        # based on your previous code snippet, you'll need to create them here too.
        # This part depends heavily on how your 'preprocessing.pkl' was created.
        # If your 'feature_names' in the pipeline are already the numerical and encoded ones, this might be redundant.
        # Assuming your `preprocessing_pipeline` handles `room_type`, `color`, `material` via simple `LabelEncoder` for direct mapping:
        
        # Reconstruct the full DataFrame for prediction, making sure columns are in the correct order
        # based on enc['feature_names'].
        # Create a dictionary to hold all feature values
        prediction_features = {}
        for feature_name in enc['feature_names']:
            if feature_name in final_df.columns:
                prediction_features[feature_name] = final_df[feature_name].iloc[0]
            elif feature_name.startswith('is_') and feature_name.replace('is_', '') in enc['room_type_encoder'].classes_:
                # Handle one-hot encoded room types if they are part of `feature_names`
                # This assumes 'is_roomname' format.
                room_name = feature_name.replace('is_', '')
                prediction_features[feature_name] = int(room_type == room_name)
            else:
                # Default for any other missing feature (should ideally not happen if feature_names are exhaustive)
                prediction_features[feature_name] = 0.0 # Or an appropriate default

        # Create a DataFrame with a single row and ensure column order
        X_predict = pd.DataFrame([prediction_features], columns=enc['feature_names'])
        
        # Ensure that the final DataFrame for prediction contains all 'feature_names' in the correct order
        dmatrix = xgb.DMatrix(X_predict.values, enable_categorical=True, feature_names=enc['feature_names'])

        # Predict probabilities
        probs = model.predict(dmatrix)[0] # Get probabilities for the single prediction

        # Get category names from the label encoder
        all_preds = [{"category": enc['category_label_encoder'].inverse_transform([i])[0], "probability": p} for i, p in enumerate(probs)]
        all_preds.sort(key=lambda x: x['probability'], reverse=True)

        top_preds = apply_domain_logic(room_type, is_seating, is_table, is_storage, all_preds, enc['category_label_encoder'])
        
        # Ensure probability values are floats, not strings, before returning
        for pred in top_preds:
            pred['probability'] = float(pred['probability'])

        return jsonify({"predicted_category": top_preds[0]['category'], "top_predictions": top_preds})

    except Exception as e:
        logging.exception("Prediction error during processing request.")
        return jsonify({"error": str(e), "message": "An error occurred during prediction. Check server logs for details."}), 500


if __name__ == '__main__':
    with app.app_context():
        load_assets()
    app.run(debug=True, host='0.0.0.0', port=5000)