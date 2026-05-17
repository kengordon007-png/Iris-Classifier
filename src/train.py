import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, classification_report

iris = load_iris()
X = iris.data  #shape (150, 4)
y = iris.target # shape(150)
print(iris.feature_names, iris.target_names) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Predictions:", y_pred[:5])
print("True labels:", y_test[:5])

accuracy  = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
from sklearn.neighbors import KNeighborsClassifier
model2 = KNeighborsClassifier(n_neighbors=5)
model2.fit(X_train, y_train)
y_pred2 = model2.predict(X_test)
print("k-NN accuracy:",accuracy_score(y_test, y_pred2))

# 1. Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names  # ['setosa', 'versicolor', 'virginica']

# 2. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Train a classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# 4. Make predictions
y_pred = clf.predict(X_test)

# 5. Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix Array:")
print(cm)

# 6. Plot the visual confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)

plt.title("Confusion Matrix for Iris Dataset")
plt.show()

# 1. Load the Iris dataset
iris = load_iris()
X = iris.data    # Features: sepal/petal dimensions
y = iris.target  # Labels: 0 (setosa), 1 (versicolor), 2 (virginica)

# 2. Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Initialize and train a classifier
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 4. Predict the species on the test data
y_pred = model.predict(X_test)

# 5. Calculate individual Precision and Recall scores
# Multi-class targets require an explicit 'average' parameter
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')

print(f"Overall Precision (Macro Avg): {precision:.4f}")
print(f"Overall Recall (Macro Avg): {recall:.4f}")
print("-" * 55)

# 6. Generate a full breakdown by target species name
print("Detailed Species-Wise Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
