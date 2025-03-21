# Brain Tumor Classification

##Project still in development

## Project Overview
This project aims to develop a deep learning-based model for brain tumor classification using X-ray images. The model is built using PyTorch and trained on preprocessed medical images to distinguish between healthy brains and those with tumors. The final trained model will be integrated into a Django web application, allowing users to upload X-ray images and receive classification results in real-time.

## Methodology

1. **Data Processing:**
   - X-ray images of the brain are collected and preprocessed (resizing, normalization, augmentation).
   - The dataset is split into training nad validation data sets.
  
    ![Not Cancer  (1)](https://github.com/user-attachments/assets/729a9ceb-c967-4462-861f-40b79d1d5879)   ![Cancer (19)](https://github.com/user-attachments/assets/5afb40e2-0f52-4bb0-b59c-6bb6318aa102)

Example of brain images. First one is healthy, while there is a tumor in second one


2. **Model Training:**
   - Initially, a Convolutional Neural Network (CNN) based on **AlexNet** is trained on the dataset.
   - Transfer learning is applied to improve performance by fine-tuning a **pretrained ResNet model**.
   - The model will be evaluated using accuracy, precision, recall, and F1-score.

3. **Deployment:**
   - The final trained model is integrated into a **Django** web application.
   - Users can upload X-ray images through the web interface.
   - The system processes the image and provides a classification result (healthy or tumor detected).
  
   ![dwadaw](https://github.com/user-attachments/assets/359c6e09-caf2-4866-9720-ddc4b7be8163)
   
 <sub>An early development site example, using early model trained on small dataset. </sub> 



## Technologies Used
- **Deep Learning:** PyTorch
- **Pretrained Models:** AlexNet, ResNet
- **Web Framework:** Django
- **Image Processing:**  PIL

## Features
- **Automated Brain Tumor Detection** using state-of-the-art CNN models.
- **User-Friendly Web Interface** for uploading and analyzing X-ray images.
- **Transfer Learning** to improve classification accuracy.
- **Secure and Scalable** backend using Django.

---
Developed by **Paweł Leszczyński**

