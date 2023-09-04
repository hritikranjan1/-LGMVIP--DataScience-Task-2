# Name: HRITIK RANJAN
# Task 2: Prediction using Decision Tree Algorithm

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.datasets import load_iris
import pydotplus
from IPython.display import Image

# Loading the iris dataset
iris = load_iris()

# Forming the iris dataframe
data = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target
y = pd.DataFrame(y, columns=["Target"])

df = pd.concat([data, y], axis=1)

# Data Visualization
sns.pairplot(df, hue="Target")
plt.show()

plt.figure(figsize=[10, 8])
sns.heatmap(df.corr(), annot=True)
plt.title("Heatmap")
plt.show()

# Splitting data into x and y (Attributes and Labels Respectively)
x = df.drop("Target", axis=1)
y = df.Target

# Performing a Train-Test Split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, random_state=123, test_size=0.2)

# Initializing the Decision Tree Model and Fitting the Data
clf_tree = DecisionTreeClassifier(random_state=123)
dt_fit = clf_tree.fit(xtrain, ytrain)

# Prediction of the Decision Tree Model
dt_predict = dt_fit.predict(xtest)

# Evaluation of the ML Model
sns.heatmap(confusion_matrix(ytest, dt_predict), annot=True)
classification_report_str = classification_report(ytest, dt_predict)
print("Classification Report:\n", classification_report_str)

accuracy = accuracy_score(ytest, dt_predict)
print("Accuracy Score:", accuracy)

# Visualize the Decision Tree
plt.figure(figsize=[15, 10])
plot_tree(clf_tree, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.show()

# Alternatively, you can also save the tree visualization as an image file
dot_data = tree.export_graphviz(clf_tree, out_file=None, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png('iris_tree.png')
Image(graph.create_png())

print("THANK YOU")

