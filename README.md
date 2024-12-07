# indian-monument-classifier-using-cvdl

Indian Heritage Classifier

This project is a comprehensive image classification system designed to identify and analyze Indian heritage monuments using deep learning. Leveraging EfficientNetB3 and a robust image augmentation pipeline, the model classifies images into one of 24 monument categories (adjustable based on dataset). Key features include confidence-based predictions, web scraping for additional information, image similarity comparisons, and dataset augmentation for balanced training.

Key Features:
Deep Learning Model: Uses EfficientNetB3 with transfer learning for high accuracy in classification.
Image Preprocessing: Includes resizing, normalization, and augmentation techniques like flipping, rotation, and zoom.
Interactive Predictions: Visualizes predictions with confidence scores and provides insights into top classes.
Web Scraping: Fetches information about predicted categories directly from Wikipedia.
Image Similarity Comparison: Identifies visually similar images for low-confidence predictions.
Dataset Augmentation: Generates additional training images for underrepresented categories.
Directory Management: Automatically detects and processes training directories.
Modular and Flexible: Easily adjustable to new datasets or categories.

How It Works:
Load and preprocess the dataset with ImageDataGenerator.
Train or load a pre-trained model for classification.
Classify uploaded images and visualize predictions.
Enhance user interaction with confidence-based additional insights and similar image retrieval.

Technologies Used:
TensorFlow & Keras for deep learning
EfficientNetB3 for transfer learning
OpenCV, PIL, and NumPy for image handling
BeautifulSoup for web scraping
Matplotlib for data visualization 

I have given 2 files one is in streamlit which shows every pre-processing and another is the Colab file for model training 
![image](https://github.com/user-attachments/assets/99a20098-9b5a-480f-ae3f-6b55f4ba6222)


Applications:
Cultural heritage preservation and analysis
Tourism and educational tools
AI-based museum guides

