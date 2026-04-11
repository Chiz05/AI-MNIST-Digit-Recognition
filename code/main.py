from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# 1️⃣ Load MNIST
print("Loading Fashion MNIST dataset...")
mnist = fetch_openml('Fashion-MNIST', version=1)
X, y = mnist.data, mnist.target

# 2️⃣ Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=10000, random_state=42
)

# 3️⃣ KNN Model
print("Training KNN...")
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
knn_acc = accuracy_score(y_test, knn_pred)

# 4️⃣ Logistic Regression
print("Training Logistic Regression...")
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
log_pred = log_reg.predict(X_test)
log_acc = accuracy_score(y_test, log_pred)

# 5️⃣ Results
print("\nRESULTS:")
print("KNN Accuracy:", knn_acc)
print("Logistic Regression Accuracy:", log_acc)

# 6️⃣ Confusion Matrix for best model
cm = confusion_matrix(y_test, knn_pred)

plt.imshow(cm)
plt.title("Confusion Matrix (KNN)")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
