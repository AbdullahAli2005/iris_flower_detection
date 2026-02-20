import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

iris = load_iris()

# create dataframes
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target # Add the species column (target)

print("Iris Dataset:")
print(df.head())

# Split the dataset into features (X) and target (y)
X = df.drop('species', axis=1) # Features (sepal length, sepal width, petal length, petal width)
y = df['species'] # Target (species: 0 = Setosa, 1 = Versicolor, 2 = Virginica)

# split dataset into training and testing set 70/30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

# make predictions on test set
y_pred = classifier.predict(X_test)

accuracy = metrics.accuracy_score(y_test, y_pred)
conf_matrix = metrics.confusion_matrix(y_test, y_pred)
class_report = metrics.classification_report(y_test, y_pred)

print(f"\nAccuracy: {accuracy * 100:.2f}%")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

plt.figure(figsize=(12, 8))
plot_tree(classifier, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title("Decision Tree for Iris Flower Classification")
plt.show()