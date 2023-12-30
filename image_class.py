# Install required libraries
# pip install opencv-python scikit-learn

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import os

# Function to extract features from images
def extract_features(image_path):
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (50, 50))
    flattened_image = resized_image.flatten()
    return flattened_image

# Load dataset
dataset_path = "path/to/dataset"
cat_images = [os.path.join(dataset_path, "cats", file) for file in os.listdir(os.path.join(dataset_path, "cats"))]
dog_images = [os.path.join(dataset_path, "dogs", file) for file in os.listdir(os.path.join(dataset_path, "dogs"))]

# Extract features and labels
cat_features = [extract_features(image) for image in cat_images]
dog_features = [extract_features(image) for image in dog_images]
features = np.concatenate([cat_features, dog_features])
labels = np.concatenate([np.zeros(len(cat_features)), np.ones(len(dog_features))])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train a k-nearest neighbors classifier
knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn_classifier.predict(X_test)

# Evaluate the performance of the classifier
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Test the classifier on a new image
new_image_path = "path/to/new/image.jpg"
new_image_features = extract_features(new_image_path)
prediction = knn_classifier.predict([new_image_features])[0]

if prediction == 0:
    print("The image contains a cat.")
else:
    print("The image contains a dog.")
