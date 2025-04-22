# Skin_Cancer_Detection

This project uses **Convolutional Neural Networks** and **DenseNet201** to classify skin lesions based on images from the **HAM10000 dataset**. It leverages **transfer learning** and image preprocessing to achieve high accuracy.

## 📊 Accuracy
- Achieved **87% accuracy**
- 7-class classification (e.g., melanoma, nevus, etc.)

## 🧠 Model Architecture
- Base model: DenseNet201
- Layers: Global Average Pooling, Dense, Dropout, Softmax
- Loss function: Categorical Crossentropy

## 🛠️ Technologies Used
- Python
- TensorFlow / Keras
- NumPy / Pandas
- Matplotlib / Seaborn

## 🖼️ Dataset
- **HAM10000**: 10,015 dermatoscopic images
- Classes: Melanocytic nevi, Melanoma, BCC, etc.

