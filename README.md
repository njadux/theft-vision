# Theft Vision: AI-Powered Shoplifting Detection  

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)  
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)  
![HuggingFace](https://img.shields.io/badge/Transformers-yellow?logo=huggingface)  
![Flask](https://img.shields.io/badge/Flask-black?logo=flask)  
![Django](https://img.shields.io/badge/Django-092E20?logo=django&logoColor=white)  
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)  

---

**Theft Vision** is a deep learning–based video analytics system for detecting theft in retail environments.  
It leverages **VideoMAE** (transformer-based video model) for temporal theft recognition and **ResNet-18** (CNN) for fast image-level detection. The system integrates with **Flask** and **Django** for deployment, enabling real-time suspicious activity monitoring.  

---

## 🏗️ System Architecture  

### 🔹 Modeling  
- **ResNet-18 (CNN)** → Image-level shoplifting detection with fast inference  
- **VideoMAE (Transformer)** → Video-based theft recognition  
- **Transfer Learning** → Fine-tuned on a custom shoplifting dataset  

### 🔹 Deployment  
- **Flask API** → Lightweight backend serving ResNet-18 predictions  
- **Django Web App** → *ShopFilter* project for delivering detection results to the frontend  

---

## ⚡ Features  
- ✅ Theft detection using **CNNs** (ResNet-18) and **Video Transformers** (VideoMAE)  
- ✅ Flask-based **REST API** for efficient deployment  
- ✅ Django-based **web interface** for shop filter detection  
- ✅ Preprocessing pipeline: *video → frames → processed .pt datasets*  
- ✅ Training logs, evaluation results, and reproducibility artifacts included  

---

## 🛠️ Tech Stack  
- **Deep Learning:** PyTorch, Hugging Face Transformers  
- **Models:** VideoMAE, ResNet-18  
- **Deployment:** Flask, Django  
- **Preprocessing:** OpenCV  
- **Dataset:** Custom shoplifting dataset  

---

## 📌 Future Work  
- Real-time video stream integration  
- Advanced anomaly detection techniques  

---

## 📂 Repository Structure  

