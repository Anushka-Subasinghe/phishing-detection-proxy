import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump

# Construct the file path to the dataset
script_dir = os.path.dirname(__file__)  # Directory of the current script
file_path = os.path.join(script_dir, 'dataset_phishing.csv')

# Load the dataset
data = pd.read_csv(file_path)

# Drop rows with null values
data.dropna(inplace=True)

# Split the data into features and target variable
X = data.drop(columns=['status'])
y = data['status']

# Identify non-numeric columns and encode categorical variables
non_numeric_columns = X.select_dtypes(exclude=['number']).columns
for column in non_numeric_columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define individual classifiers
classifiers = [
    ('Random Forest', RandomForestClassifier(n_estimators=100, max_depth=None)),
    ('KNN', KNeighborsClassifier(n_neighbors=5)),
    ('Gradient Boosting', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)),
    ('Logistic Regression', LogisticRegression(max_iter=1000, C=1, solver='liblinear'))
]

# Create a Voting Classifier
voting_classifier = VotingClassifier(estimators=classifiers, voting='hard')

# Train the Voting Classifier
voting_classifier.fit(X_train, y_train)

# Evaluate the Voting Classifier
y_pred = voting_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Voting Classifier Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the Voting Classifier
model_filename = os.path.join(script_dir, 'voting_classifier_model.joblib')
dump(voting_classifier, model_filename)
