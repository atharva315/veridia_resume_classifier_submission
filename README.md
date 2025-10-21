Here's a **formal and professional README** for your complete project:

---

# Resume Classification System Using BERT

[![Python](https://img.shields.iottps://img.shieldsps://img.shields.io/badge/Transformers-4.35+-yellow.io/badge/License-MIT-light end-to-end AI-powered resume classification system leveraging BERT for automated candidate screening across 24 job categories.**

**Developed for:** Veridia.io AI/ML Internship  
**Author:** Atharva Gaikwad  
**Date:** October 2025

***

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Performance Metrics](#performance-metrics)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Screenshots](#screenshots)
- [Documentation](#documentation)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

***

## 🎯 Overview

The **Resume Classification System** is a machine learning solution designed to automate the resume screening process for recruitment workflows. Using state-of-the-art Natural Language Processing (NLP) with BERT (Bidirectional Encoder Representations from Transformers), the system accurately categorizes resumes into 24 distinct job roles, significantly reducing manual effort and improving hiring efficiency.

### Problem Statement

Manual resume screening is:
- **Time-consuming**: HR teams spend hours reviewing hundreds of applications
- **Subjective**: Human bias can affect candidate evaluation
- **Inconsistent**: Different reviewers may categorize the same resume differently

### Solution

An automated classification system that:
- Processes resumes in PDF, DOCX, and TXT formats
- Provides instant categorization with confidence scores
- Achieves 87.15% accuracy through deep learning
- Offers an intuitive web interface for easy deployment

***

## ✨ Features

- **Multi-format Support**: Accepts PDF, DOCX, and TXT resume files
- **24 Job Categories**: Comprehensive classification across diverse professional domains
- **Real-time Inference**: Instant predictions with confidence scores
- **Interactive Web Interface**: User-friendly Streamlit application
- **Comprehensive Metrics**: Detailed performance evaluation with visualizations
- **GPU Acceleration**: Optimized training with mixed precision and gradient accumulation
- **Early Stopping**: Prevents overfitting with validation-based checkpointing
- **Reproducible**: Complete code with documentation for easy replication

***

## 📊 Performance Metrics

| Metric | Score |
|--------|-------|
| **Test Accuracy** | 88.35% |
| **Weighted F1-Score** | 87.56% |
| **Precision** | 87.63% |
| **Recall** | 88.35% |
| **Training Time** | ~14 minutes (Tesla T4 GPU) |
| **Total Parameters** | 110M (BERT-base) |

### Top-Performing Categories

- **Java Developer**: 92% F1-Score
- **Data Science**: 89% F1-Score
- **HR**: 87% F1-Score

***

## 🛠️ Technology Stack

### Core Frameworks
- **Deep Learning**: PyTorch 2.0+
- **NLP**: Hugging Face Transformers 4.35+
- **Pre-trained Model**: BERT (bert-base-uncased)

### Libraries & Tools
- **Data Processing**: Pandas, NumPy, scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Text Extraction**: pdfplumber, python-docx
- **Web Framework**: Streamlit
- **Training Platform**: Google Colab (T4 GPU)
- **Deployment**: ngrok / localtunnel

***

## 📁 Project Structure

```
Veridia_Resume_Classifier_Submission/
│
├── code/
│   ├── Resume_Classifier.ipynb       # Complete training notebook
│   ├── app.py                         # Streamlit web application
│   └── requirements.txt               # Python dependencies
│
├── model/
│   └── resume_classifier_best/        # Trained BERT model files
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer_config.json
│       ├── vocab.txt
│       └── special_tokens_map.json
│
├── outputs/
│   ├── category_distribution.png      # Dataset visualization
│   ├── confusion_matrix.png           # Model evaluation
│   ├── training_history.png           # Training curves
│   ├── label_encoder.pkl              # Category encoder
│   └── classification_report.txt      # Detailed metrics
│
├── Project report/
│   └── PROJECT_REPORT.pdf             # Comprehensive documentation
│
├── result screenshot/
│   └── [Application demo screenshots]
│
├── kaggle.json                        # Kaggle API credentials                         
└── HOW TO RUN ON ANY DEVICE.docx     # Detailed execution instructions
```

***

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (optional but recommended)
- Google Colab account (for training)
- Kaggle account (for dataset access)

### Quick Start

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/veridia-resume-classifier.git
   cd veridia-resume-classifier
   ```

2. **Install Dependencies**
   ```bash
   pip install -r code/requirements.txt
   ```

3. **Download Pre-trained Model**
   - The trained model is available in the `model/resume_classifier_best/` directory
   - Alternatively, download from [Google Drive](https://drive.google.com/drive/folders/1h7_Ypjlhb9SQTOakE84VbdX2hF-S-JA1)

4. **Run the Application**
   ```bash
   streamlit run code/app.py
   ```

***

## 💻 Usage

### Training the Model (Google Colab)

1. Open `code/Resume_Classifier.ipynb` in Google Colab
2. Upload your `kaggle.json` when prompted
3. Run all cells sequentially
4. The trained model will be saved to Google Drive

### Running the Web Application

1. Ensure the model is in the correct directory
2. Launch Streamlit:
   ```bash
   streamlit run code/app.py
   ```
3. Access the application at `http://localhost:8501`
4. Upload a resume file and view predictions

### Using the Deployed App

For live demos, use ngrok or localtunnel:
```bash
ngrok http 8501
```
Share the generated public URL for remote access.

***

## 🧠 Model Architecture

### Pipeline Overview

```
Input Resume → Text Extraction → Preprocessing → BERT Tokenization →
→ BERT Encoder → Classification Head → Softmax → Category Prediction
```

### Technical Details

- **Base Model**: BERT-base-uncased (12 layers, 768 hidden units)
- **Classification Head**: Linear layer (768 → 24 classes)
- **Loss Function**: Cross-entropy
- **Optimizer**: AdamW (lr=3e-5, weight_decay=0.01)
- **Scheduler**: Cosine with warmup (10% warmup ratio)
- **Batch Size**: 16 with gradient accumulation (effective batch size: 32)
- **Mixed Precision**: FP16 for faster training
- **Early Stopping**: Patience = 3 epochs

### Data Preprocessing

1. **Text Cleaning**:
   - Remove URLs, emails, HTML tags, special characters
   - Convert to lowercase
   - Filter resumes <50 characters

2. **Tokenization**:
   - BERT WordPiece tokenizer
   - Max length: 512 tokens
   - Padding and truncation applied

3. **Label Encoding**:
   - 24 job categories mapped to integer labels
   - Stratified train-validation-test split (80-10-10)

***

## 📈 Results

### Confusion Matrix



### Training History

![Training Curves](outputs/training_history.pngution](outputs/category_distribution. Report

Detailed per-class metrics available in `outputs/classification_report.txt`

***

## 📸 Screenshots

### Streamlit Web Interface

![App Interface](result_screenshot/app_demo Results

![Classification Output](result_screenshot/results 📚 Documentation

- **[PROJECT_REPORT.pdf](Project%20report/PROJECT_REPORT.pdf)**: Comprehensive project documentation
- **[HOW TO RUN ON ANY DEVICE.docx](HOW%20TO%20RUN%20ON%20ANY%20DEVICE.docx)**: Step-by-step execution guide
  

***

## 🔮 Future Enhancements

- **Multi-language Support**: Extend to non-English resumes
- **Skill Extraction**: Extract key skills and technologies from resumes
- **Experience Detection**: Identify years of experience and seniority level
- **Job Matching**: Recommend suitable job postings based on resume content
- **Permanent Deployment**: Host on AWS/GCP/Azure for 24/7 availability
- **API Development**: RESTful API for integration with ATS systems
- **Batch Processing**: Handle multiple resume uploads simultaneously

***

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

***

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

***

## 📧 Contact

**Atharva Gaikwad**  
- GitHub: [@atharva315](https://github.com/atharva315)
- Email: gaikwadatharva315@example.com
- LinkedIn: [LinkedIn](https://www.linkedin.com/in/atharva-gaikwad-05493b282)

**Project Link**: [https://github.com/atharva315/veridia-resume-classifier](https://github.com/atharva315/veridia-resume-classifier)

***

## 🙏 Acknowledgments

- **Veridia.io** for the internship opportunity and project guidance
- **Hugging Face** for the Transformers library and pre-trained models
- **Google Colab** for providing free GPU resources
- **Kaggle** for hosting the resume dataset
- **Streamlit** for the intuitive web framework

***

## 📊 Dataset

**Source**: [Kaggle Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset)  
**Samples**: 2,484 resumes  
**Categories**: 24 job roles  
**Format**: CSV with resume text and category labels

***

## 🔗 Additional Resources

- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [Streamlit Documentation](https://docs.streamlit.io)
- [PyTorch Tutorials](https://pytorch.org/tutorials)

***

<div align="center">

**⭐ If you find this project useful, please consider giving it a star! ⭐**

Made with ❤️ by Atharva Gaikwad

</div>

***


