from sklearn.metrics import accuracy_score

def load_and_train_model(filename):
    # Try reading the file with different encodings
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

    print("Columns in the CSV file:", data.columns.tolist())

    # Preprocess data
    data['Nutrient Level'] = data['Nutrient Level'].map({'Poor': 0, 'Good': 1, 'Excellent': 2})

    # Feature selection
    X = data[['Temperature (°C)', 'pH', 'OD (mg/L)', 'TDS (ppt)', 'Salinity (ppt)']]
    y = data['Nutrient Level']

    # Splitting data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Training model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # **Calculate accuracy**
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")  # Print accuracy to console

    return model

# Load the model and display accuracy
model = load_and_train_model('Shrimp.csv')
