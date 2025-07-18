from flask import Flask, request, jsonify
from flask_cors import CORS
from joblib import load
import logging
import json
import os
from datetime import datetime
from joblib import load

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Enable CORS for all domains on all routes
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Global variables to store model and data
model = None
df = None
MEASUREMENTS_FILE = 'measurements.json'

def initialize_model():
    global model
    try:
        model = load("swakriti_body_predictor.pkl")  # Your actual model
        logger.info("Pre-trained model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")

def load_measurements():
    """Load measurements from JSON file."""
    if os.path.exists(MEASUREMENTS_FILE):
        try:
            with open(MEASUREMENTS_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Could not decode {MEASUREMENTS_FILE}. Starting with empty data.")
            return {}
    return {}

def save_measurements(measurements_data):
    """Save measurements to JSON file."""
    try:
        with open(MEASUREMENTS_FILE, 'w') as f:
            json.dump(measurements_data, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving measurements: {str(e)}")
        return False
    
def calculate_additional_measurements(age, gender_str, height):
    """Calculate Chest, Shoulder, and Sleeve using formulas based on age, gender, and height."""
    if age < 2:
        chest = height * 0.51
    elif age < 6:
        chest = height * 0.49
    else:
        chest = height * 0.47

    if gender_str == "male":
        shoulder = height * (0.22 if age < 6 else 0.23)
    else:
        shoulder = height * (0.21 if age < 6 else 0.22)

    if age < 2:
        sleeve = height * 0.28
    elif age < 6:
        sleeve = height * 0.30
    else:
        sleeve = height * 0.32

    return chest, shoulder, sleeve

def validate_input(data, for_update=False):
    required_fields = ['parent_id', 'child_id']
    if not for_update:
        required_fields.extend(['height', 'weight', 'gender', 'age'])

    for field in required_fields:
        if field not in data:
            return False, f"Missing required field: {field}"

    if not isinstance(data['parent_id'], str) or not data['parent_id'].strip():
        return False, "Parent ID must be a non-empty string"
    if not isinstance(data['child_id'], str) or not data['child_id'].strip():
        return False, "Child ID must be a non-empty string"

    # Skip other validations for update requests with only measurements
    if for_update and len(data) == 2 and 'measurements' in data:
        return validate_measurements_format(data['measurements'])
    
    if not for_update:
        # Validate data types and ranges
        try:
            age = float(data['age'])
            if not (3 <= age <= 18):
                return False, "Age must be between 3 and 18 years"
        except (ValueError, TypeError):
            return False, "Age must be a valid number"
        
        try:
            weight = float(data['weight'])
            if not (10.0 <= weight <= 120.0):
                return False, "Weight must be between 10.0 and 120.0 kg"
        except (ValueError, TypeError):
            return False, "Weight must be a valid number"
        
        try:
            height = float(data['height'])
            if not (80.0 <= height <= 220.0):
                return False, "Height must be between 80.0 and 220.0 cm"
        except (ValueError, TypeError):
            return False, "Height must be a valid number"
        
        # Validate gender
        gender = data['gender']
        if isinstance(gender, str):
            gender_lower = gender.lower()
            if gender_lower not in ['male', 'female', 'm', 'f']:
                return False, "Gender must be 'male', 'female', 'm', or 'f'"
        elif isinstance(gender, int):
            if gender not in [1, 2]:
                return False, "Gender must be 1 (male) or 2 (female)"
        else:
            return False, "Gender must be a string or integer"
    
    return True, "Valid"

def validate_measurements_format(measurements):
    """Validate measurements format for updates."""
    if not isinstance(measurements, dict):
        return False, "Measurements must be a dictionary"
    
    valid_measurement_keys = ['waist', 'hip', 'bicep', 'neck', 'wrist', 'chest', 'shoulder', 'sleeve']

    for key, value in measurements.items():
        if key not in valid_measurement_keys:
            return False, f"Invalid measurement key: {key}. Valid keys are: {', '.join(valid_measurement_keys)}"
        
        try:
            float_value = float(value)
            if float_value <= 0:
                return False, f"Measurement {key} must be a positive number"
        except (ValueError, TypeError):
            return False, f"Measurement {key} must be a valid number"
    
    return True, "Valid"

def convert_gender(gender):
    """Convert gender to numeric format."""
    if isinstance(gender, str):
        gender_lower = gender.lower()
        if gender_lower in ['male', 'm']:
            return 1
        elif gender_lower in ['female', 'f']:
            return 2
    elif isinstance(gender, int):
        return gender
    return 1  # Default to male

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    measurements_data = load_measurements()
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'total_users': len(measurements_data),
        'measurements_file': MEASUREMENTS_FILE
    })

@app.route('/predict', methods=['POST'])
def predict_measurements():
    """Predict body measurements based on input parameters."""
    try:
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400

        data = request.get_json()
        is_valid, message = validate_input(data)
        if not is_valid:
            return jsonify({'error': message}), 400

        if model is None:
            return jsonify({'error': 'Model not initialized'}), 500

        parent_id = str(data['parent_id'])
        child_id = str(data['child_id'])

        age = float(data['age'])
        gender = convert_gender(data['gender'])
        weight = float(data['weight'])
        height = float(data['height'])
        input_data = [[age, gender, weight, height]]

        prediction = model.predict(input_data)[0]
        waist, hip, bicep, neck, wrist = prediction[:5]
        chest, shoulder, sleeve = calculate_additional_measurements(age, 'male' if gender == 1 else 'female', height)

        measurements = {
            'waist': float(round(waist, 2)),
            'hip': float(round(hip, 2)),
            'bicep': float(round(bicep, 2)),
            'neck': float(round(neck, 2)),
            'wrist': float(round(wrist, 2)),
            'chest': float(round(chest, 2)),
            'shoulder': float(round(shoulder, 2)),
            'sleeve': float(round(sleeve, 2))
        }

        # Load and structure data
        measurements_data = load_measurements()
        if parent_id not in measurements_data:
            measurements_data[parent_id] = {}

        user_data = {
            'parent_id': parent_id,
            'child_id': child_id,
            'input_parameters': {
                'age': age,
                'gender': 'male' if gender == 1 else 'female',
                'weight': weight,
                'height': height
            },
            'measurements_cm': measurements,
            'measurements_inches': {
                key: round(value / 2.54, 2) for key, value in measurements.items()
            },
            'prediction_timestamp': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'is_predicted': True,
            'is_manually_updated': False
        }

        measurements_data[parent_id][child_id] = user_data

        if save_measurements(measurements_data):
            logger.info(f"Measurements saved for {parent_id}/{child_id}")
        else:
            logger.error(f"Failed to save measurements for {parent_id}/{child_id}")

        return jsonify({
            'success': True,
            'parent_id': parent_id,
            'child_id': child_id,
            'measurements_cm': measurements,
            'measurements_inches': {
                key: round(value / 2.54, 2) for key, value in measurements.items()
            },
            'message': 'Measurements predicted and saved successfully'
        })

    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/update-measurements', methods=['PUT'])
def update_measurements():
    """Update measurements for a specific child under a parent."""
    try:
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400

        data = request.get_json()

        is_valid, message = validate_input(data, for_update=True)
        if not is_valid:
            return jsonify({'error': message}), 400

        parent_id = str(data['parent_id'])
        child_id = str(data['child_id'])

        measurements_data = load_measurements()

        if parent_id not in measurements_data or child_id not in measurements_data[parent_id]:
            return jsonify({'error': f'Child {child_id} under parent {parent_id} not found. Please make a prediction first.'}), 404

        user_data = measurements_data[parent_id][child_id]

        if 'measurements' in data:
            is_valid, message = validate_measurements_format(data['measurements'])
            if not is_valid:
                return jsonify({'error': message}), 400

            for key, value in data['measurements'].items():
                user_data['measurements_cm'][key] = round(float(value), 2)

            user_data['measurements_inches'] = {
                key: round(value / 2.54, 2) for key, value in user_data['measurements_cm'].items()
            }

            user_data['last_updated'] = datetime.now().isoformat()
            user_data['is_manually_updated'] = True

        if save_measurements(measurements_data):
            logger.info(f"Measurements updated for {parent_id}/{child_id}")
        else:
            logger.error(f"Failed to update measurements for {parent_id}/{child_id}")
            return jsonify({'error': 'Failed to save updated measurements'}), 500

        return jsonify({
            'success': True,
            'parent_id': parent_id,
            'child_id': child_id,
            'measurements_cm': user_data['measurements_cm'],
            'measurements_inches': user_data['measurements_inches'],
            'message': 'Measurements updated successfully'
        })

    except Exception as e:
        logger.error(f"Error updating measurements: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/get-measurements/<parent_id>/<child_id>', methods=['GET'])
def get_measurements(parent_id, child_id):
    """Get measurements for a specific child under a parent."""
    try:
        measurements_data = load_measurements()

        if parent_id not in measurements_data or child_id not in measurements_data[parent_id]:
            return jsonify({'error': f'Child {child_id} under parent {parent_id} not found'}), 404

        user_data = measurements_data[parent_id][child_id]

        return jsonify({
            'success': True,
            'parent_id': parent_id,
            'child_id': child_id,
            'input_parameters': user_data.get('input_parameters', {}),
            'measurements_cm': user_data.get('measurements_cm', {}),
            'measurements_inches': user_data.get('measurements_inches', {}),
            'prediction_timestamp': user_data.get('prediction_timestamp', ''),
            'last_updated': user_data.get('last_updated', ''),
            'is_predicted': user_data.get('is_predicted', False),
            'is_manually_updated': user_data.get('is_manually_updated', False)
        })

    except Exception as e:
        logger.error(f"Error retrieving measurements: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({'error': 'Method not allowed'}), 405

# Initialize model on startup
initialize_model()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)