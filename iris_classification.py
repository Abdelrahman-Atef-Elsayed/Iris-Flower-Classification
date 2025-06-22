import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# تحميل البيانات
iris = load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names

# تقسيم البيانات للتجريب النهائي بعد الـ cross-validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# تعريف النماذج
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier()
}

# تجربة Cross-Validation واختيار أفضل موديل
cv_results = {}
print("Cross-Validation Results:")
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5)
    mean_score = np.mean(scores)
    cv_results[name] = mean_score
    print(f"{name}: Mean Accuracy = {mean_score:.2f}")

# اختيار أفضل موديل
best_model_name = max(cv_results, key=cv_results.get)
best_model = models[best_model_name]
print(f"\n✅ Best Model: {best_model_name} (Accuracy: {cv_results[best_model_name]:.2f})")

# تدريب أفضل موديل على البيانات المقسّمة
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

# عرض النتائج النهائية
print(f"\nFinal Accuracy on Test Set: {accuracy_score(y_test, y_pred):.2f}")

# مصفوفة الالتباس
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, cmap='YlGnBu', xticklabels=class_names, yticklabels=class_names, fmt='d')
plt.title(f'Confusion Matrix - {best_model_name}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()
