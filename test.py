import tensorflow
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import ResNet50, preprocess_input
from tensorflow.keras.layers import MaxPooling2D
import cv2
import numpy as np
from numpy.linalg import norm
import pickle
from sklearn.neighbors import NearestNeighbors
import os

# Load precomputed features and filenames
feature_list = np.array(pickle.load(open('Featurevector.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Load the ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# Create the model with MaxPooling
model = tensorflow.keras.Sequential([
    base_model,
    MaxPooling2D()
])
model.summary()

# Load and preprocess the input image
img_path = '9980.jpg'  # Replace with your image path
if not os.path.exists(img_path):
    print(f"Error: Image file '{img_path}' not found.")
else:
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = np.array(img)
    expand_img = np.expand_dims(img, axis=0)
    pre_img = preprocess_input(expand_img)

    # Extract features and normalize
    result = model.predict(pre_img).flatten()
    normalized = result / norm(result)

    # Fit NearestNeighbors and find similar images
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([normalized])

    print("Indices of similar images:", indices)
    print("Distances:", distances)

    # Display filenames of similar images
    for file_idx in indices[0][1:6]:  # Skip the first index as it will be the input image itself
        print(filenames[file_idx])
