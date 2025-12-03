# Fine-Tuning de Mask R-CNN y Faster R-CNN con Torchvision  
Entrenamiento, evaluación y visualización de detecciones usando el dataset Penn-Fudan

---

## 🧩 Descripción del Proyecto

Este repositorio muestra cómo **entrenar y ajustar modelos de detección y segmentación de instancias** usando PyTorch y Torchvision, específicamente:

- **Faster R-CNN**
- **Mask R-CNN**

El proyecto utiliza el dataset **Penn-Fudan Pedestrian**, que contiene imágenes urbanas con personas y sus máscaras de segmentación.

Se cubre el proceso completo:
1. Carga del dataset  
2. Construcción de un `Dataset` personalizado  
3. Aplicación de transformaciones  
4. Fine-tuning de Mask R-CNN  
5. Entrenamiento y evaluación  
6. Visualización de bounding boxes y máscaras  

Este flujo sigue la referencia oficial de Torchvision:
https://github.com/pytorch/vision/tree/main/references/detection

---

🧩 Requisitos

Antes de ejecutar el script, instala las dependencias:

pip install -r requirements.txt

🧑‍💻 Autor

Desarrollado por Gus como parte de su aprendizaje en Python e IA.
