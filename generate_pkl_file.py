import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression

# Dummy data (3 samples, 2 features)
X = np.array([[1.2, 3.4], [2.5, 1.9], [0.8, 2.2]])
y = np.array([0, 1, 0])  # Labels (binary classification)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X, y)

# Save the model as model.pkl
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Simple model saved as model.pkl!")
