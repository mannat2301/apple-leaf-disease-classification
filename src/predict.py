import cv2
import joblib
import numpy as np

model = joblib.load("models/random_forest_model.pkl")

classes = [
    "Apple Scab",
    "Black Rot",
    "Cedar Apple Rust",
    "Healthy"
]

def predict(image_path):

    img = cv2.imread(image_path)
    img = cv2.resize(img,(64,64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.flatten().reshape(1,-1)

    pred = model.predict(img)

    return classes[pred[0]]

result = predict("test_leaf.jpg")

print("Prediction:", result)
