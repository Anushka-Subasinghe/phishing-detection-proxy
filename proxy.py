import os
from mitmproxy import http
import pandas as pd
import joblib

# Directory of the current script
script_dir = os.path.dirname(__file__)

# File path for the model
model_file = 'rf_classifier_model.joblib'
model_path = os.path.join(script_dir, model_file)

# Load the pre-trained Random Forest classifier model
model = joblib.load(model_path)

# Function to preprocess URL features
def preprocess_url(url):
    # Extract features from the URL
    features = {
        'length_url': len(url),
        # Add more features as needed
    }
    return pd.DataFrame([features])

# Function to analyze URLs and make predictions
def analyze_url(url):
    # Preprocess URL features
    url_features = preprocess_url(url)
    
    # Make predictions using the pre-trained model
    prediction = model.predict(url_features)[0]
    if prediction == 1:
        return "Phishing"
    else:
        return "Legitimate"

# Event handler for intercepted HTTP requests
def request(flow: http.HTTPFlow) -> None:
    # Extract the URL from the request
    url = flow.request.url
    
    # Analyze the URL
    result = analyze_url(url)
    
    # Log the analysis result
    print(f"URL: {url}, Analysis Result: {result}")
    
    # Example action based on the analysis result
    if result == "Phishing":
        # Block access to the phishing URL
        flow.response = http.HTTPResponse.make(403, b"Forbidden")
        print("Access blocked for phishing URL:", url)
    else:
        # Allow access to legitimate URLs
        print("Access allowed for legitimate URL:", url)

# Start the proxy server
def start():
    from mitmproxy.tools.main import mitmdump
    mitmdump(['-s', __file__])

# Main entry point
if __name__ == '__main__':
    start()
