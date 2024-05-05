import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
dataset = pickle.load(open('sign_language_dataset.pickle', 'rb'))

# Validate consistent feature counts
data = np.asarray([item for item in dataset['data'] if len(item) == min(set(len(x) for x in dataset['data']))])
labels = np.asarray([label for i, label in enumerate(dataset['labels']) if len(dataset['data'][i]) == len(data[0])])

# Ensure all data has the same number of features
assert all(len(x) == len(data[0]) for x in data), "Inconsistent feature counts in the data"

# Split into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, stratify=labels)

# Train a Random Forest classifier
model = RandomForestClassifier(n_estimators=100)  # Using 100 trees
model.fit(x_train, y_train)

# Evaluate the classifier
y_predict = model.predict(x_test)
accuracy = accuracy_score(y_predict, y_test)

print(f'Accuracy: {accuracy * 100:.2f}%')
print('Classification Report:')
print(classification_report(y_test, y_predict))

# Save the trained model
with open('trained_model2.pickle', 'wb') as f:
    pickle.dump({'model': model}, f)
