# CNN Pill Defect Classification & Localization
<img width="2256" height="1504" alt="Screenshot 2025-09-18 173326" src="https://github.com/user-attachments/assets/ae037136-78e1-4c0b-abd4-1214431b6889" />
<img width="1295" height="487" alt="image" src="https://github.com/user-attachments/assets/550a1326-dafb-4bb1-aa86-c75d6566fd9f" />
https://colab.research.google.com/drive/1bjRcLtD6M4ANVpa76zbGMBg5W2svmjSt?usp=sharing

## Description
This project implements a Convolutional Neural Network (CNN) to automatically classify and localize defects in pharmaceutical pills. The system not only determines whether a pill is normal, cracked, or broken but also highlights the defect area using bounding boxes. The model has been optimized and exported to TensorFlow Lite for deployment on embedded or mobile devices.

## Features
- Pill classification into three categories: **Normal**, **Cracked**, and **Broken**  
- Defect localization using bounding box detection  
- Lightweight model with TensorFlow Lite support  
- Visualization of predictions with overlays  

## Methodology
- **Architecture**: MobileNetV2-based CNN fine-tuned for defect classification and localization  
- **Dataset**: Labeled images of pills with three defect classes  
- **Preprocessing & Augmentation**: Image resizing, normalization, rotation, flipping, and brightness adjustments  
- **Output**: Class label + bounding box coordinates for defect localization  
