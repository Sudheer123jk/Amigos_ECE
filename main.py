import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Function to train and evaluate the model
def train_and_evaluate(data):
    # Display column names for debugging
    print("Columns in the dataset:", data.columns.tolist())

    # Convert target variable to numerical values
    data['Nutrient Level'] = data['Nutrient Level'].map({'Poor': 0, 'Good': 1, 'Excellent': 2})

    # Selecting feature columns
    X = data[['Temperature (°C)', 'pH', 'OD (mg/L)', 'TDS (ppt)', 'Salinity (ppt)']]
    y = data['Nutrient Level']

    # Splitting dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initializing and training the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Making predictions
    y_pred = model.predict(X_test)

    # Calculating accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")

    return model


# Load dataset dynamically
def load_dataset(filename):
    # Try different encodings
    encodings = ['utf-8', 'windows-1252']
    data = None

    for encoding in encodings:
        try:
            data = pd.read_csv(filename, encoding=encoding)
            print(f"File successfully read with encoding: {encoding}")
            break
        except UnicodeDecodeError:
            print(f"Failed to read with encoding: {encoding}")
            continue

    if data is None:
        raise ValueError("Unable to read the file with any of the specified encodings.")

    return data


# Example: Pass dataset dynamically
dataset = load_dataset('Shrimp.csv')  # Replace with your file path
model = train_and_evaluate(dataset)
