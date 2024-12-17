Overview
This project implements a deep learning-based teledermatology system for classifying various skin diseases using images. Leveraging the power of transfer learning with the VGG16 architecture, this model can classify 23 distinct skin conditions with high accuracy. It aims to serve as a diagnostic aid for dermatologists and provide early disease detection for patients in remote areas, improving healthcare accessibility.

Features
23 Skin Disease Categories: The model can classify a range of skin conditions, including eczema, psoriasis, acne, melanoma, and more.
Data Augmentation: Handles class imbalance with rotation, zoom, flipping, and shifting techniques.
Performance Metrics: Achieves competitive accuracy, precision, recall, and F1-score across all classes.
Pre-trained Model: Utilizes the VGG16 model, pre-trained on ImageNet, for feature extraction and classification.

Dataset
The dataset consists of labeled images of skin diseases sourced from publicly available datasets:
DermNet

Data Preprocessing:
Images resized to 128x128 pixels for compatibility with the model.
Pixel values normalized to the range [0, 1].
Data augmentation techniques applied to address class imbalance.

Architecture
The model is built on the VGG16 architecture:

Base Model: Pre-trained VGG16 layers for feature extraction.

Custom Layers:
Global Average Pooling (GAP) for dimensionality reduction.
Fully connected layer with 1024 neurons and ReLU activation.
Dropout for regularization to prevent overfitting.
Softmax output layer for multi-class classification (23 categories).

Performance
The model was evaluated on key metrics:
Accuracy: 92.5% (Proposed Model)
Precision: 90.2%
Recall: 88.7%
F1-Score: 89.4%
Graphs and confusion matrices were generated to visualize training performance and prediction accuracy.

Future Work
Dataset Expansion: Incorporate larger and more diverse datasets to improve model robustness.
Advanced Architectures: Explore newer architectures like EfficientNet or Vision Transformers for better performance.
Explainable AI: Integrate tools like Grad-CAM for visualization of the model’s decision-making process.
Mobile Optimization: Compress and deploy the model on mobile platforms for real-time diagnosis.
Telemedicine Integration: Extend the project to include a full-fledged telemedicine platform.

Contributors: Shiksha SK

License:
This project is licensed under the Apache License.

Acknowledgments
Special thanks to:
The creators of ISIC Archive and DermNet for providing valuable datasets.
The TensorFlow and Keras teams for their powerful tools and libraries.
