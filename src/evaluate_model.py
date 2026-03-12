import joblib
from sklearn.metrics import classification_report
from preprocess import load_images
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

X, y = load_images("data/raw")

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = joblib.load("models/random_forest_model.pkl")

predictions = model.predict(X_test)

print(classification_report(y_test, predictions))
