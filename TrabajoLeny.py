import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Cargar dataset
iris = load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names

# 2. Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Entrenar modelos (One-vs-Rest con regresión lineal)
models = []
for clase in np.unique(y):
    y_train_bin = (y_train == clase).astype(int)
    model = LinearRegression()
    model.fit(X_train, y_train_bin)
    models.append(model)

# 4. Predicción multiclase
def predict_multiclass(X):
    preds = np.column_stack([m.predict(X) for m in models])
    return np.argmax(preds, axis=1)

y_pred = predict_multiclass(X_test)

# 5. Métricas
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)
print("\nReporte de Clasificación:\n", classification_report(y_test, y_pred, target_names=class_names))

# 6. Matriz de confusión
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="PuRd",
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de Confusión")
plt.show()
