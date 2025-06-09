from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# تحميل بيانات Iris
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)

# تدريب نموذج Decision Tree
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# التنبؤ والتقييم
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
