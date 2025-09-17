# Clasificación del Dataset Iris utilizando Regresión Lineal

Este repositorio contiene un experimento de **clasificación multiclase** sobre el famoso dataset **Iris**, empleando un enfoque no convencional: **Regresión Lineal** con técnica *One-vs-Rest (OvR)*.  

---

## 📂 Contenido del Repositorio
- `TrabajoLeny.py`: Script en Python que implementa el modelo de clasificación utilizando Scikit-Learn, con visualización de la matriz de confusión.
- `Informe de Clasificación del Dataset Iris utilizando Regresión Lineal.docx`: Documento que explica la metodología, resultados y conclusiones del experimento.

---

## ⚙️ Metodología
1. **Dataset:**  
   - Iris (150 muestras, 3 clases: Setosa, Versicolor, Virginica).  
   - 4 características: longitud y ancho del sépalo, longitud y ancho del pétalo.  

2. **Preprocesamiento:**  
   - División en 70% entrenamiento y 30% prueba.  

3. **Modelo:**  
   - Estrategia **One-vs-Rest (OvR)**: se entrena un modelo de regresión lineal para cada clase.  
   - Predicción multiclase mediante selección de la clase con mayor salida (*argmax*).  

4. **Evaluación:**  
   - Métricas: Accuracy, Precision, Recall, F1-score.  
   - Matriz de confusión con visualización gráfica.  

---

## 📊 Resultados Principales
- **Accuracy global:** 82.2%  
- **Setosa:** identificada perfectamente (Precisión = Recall = F1 = 1.00).  
- **Versicolor:** Precisión = 0.86, Recall = 0.46 (confusión con Virginica).  
- **Virginica:** Precisión = 0.63, Recall = 0.92.  

La matriz de confusión muestra una clara confusión entre Versicolor y Virginica, mientras que Setosa se clasifica sin errores.  

---

## 📝 Conclusiones
- El uso de **Regresión Lineal** para clasificación ofrece un desempeño razonable (>80%), pero no es ideal para separar todas las clases.  
- El modelo distingue perfectamente **Setosa**, pero presenta dificultades para diferenciar **Versicolor** y **Virginica**.  
- Esto evidencia las limitaciones del uso de regresión lineal en clasificación y motiva a utilizar algoritmos más adecuados, como la **Regresión Logística**.  

---

## 🚀 Requisitos para ejecutar el código
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
python TrabajoLeny.py
