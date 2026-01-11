from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# ---- Load model and encoder ----
try:
    model = joblib.load("fashion_model.pkl")
    encoder = joblib.load("channel_encoder.pkl")
except Exception as e:
    raise RuntimeError(f"Error loading model or encoder: {e}")

# ---- Home route ----
@app.route('/')
def home():
    return render_template('index.html')

# ---- Prediction route ----
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        discount = float(request.form['discount'])
        duration = int(request.form['duration'])
        channel = request.form['channel']

        # Handle encoding
        if hasattr(encoder, "transform"):
            try:
                encoded_value = encoder.transform([[channel]]) if len(encoder.transform.__code__.co_varnames) > 1 else encoder.transform([channel])
                if hasattr(encoded_value, "toarray"):
                    encoded_value = encoded_value.toarray()[0]
                channel_encoded = encoded_value[0] if isinstance(encoded_value, np.ndarray) else encoded_value
            except Exception:
                # Fallback for LabelEncoder
                channel_encoded = encoder.transform([channel])[0]
        else:
            return render_template('index.html', result="❌ Encoder missing 'transform' method.")

        # Prepare input dataframe
        input_df = pd.DataFrame({
            "discount_numeric": [discount],
            "duration_days": [duration],
            "channel_encoded": [channel_encoded]
        })

        # Make prediction
        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        # Prepare output
        if prediction == 1:
            result = f"✅ Campaign is likely to be successful! (Confidence: {prob:.2%})"
        else:
            result = f"⚠️ Campaign might not succeed. (Confidence: {prob:.2%})"

        return render_template('index.html', result=result)

    except Exception as e:
        return render_template('index.html', result=f"Error: {str(e)}")

# ---- Run app ----
if __name__ == "__main__":
    app.run(debug=True)
