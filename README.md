# Bangla News Article Classification using Deep Learning

## 🚀 Project Overview  

This project focuses on building **deep learning models** to classify Bangla text into predefined categories. Leveraging GRU, LSTM, and CNN architectures, it achieves high accuracy while handling large-scale textual datasets.  

The project is designed to streamline text classification in Bangla, providing valuable insights for various applications like sentiment analysis, news categorization, and more.

---

## 📂 Key Features  

- **Preprocessing Pipeline**: Efficient text tokenization and padding for uniform sequence lengths.
- **Multi-Model Implementation**: Three distinct architectures (GRU, LSTM, CNN) implemented for performance comparison.
- **Class Imbalance Handling**: Utilizes class weights for balanced training across categories.
- **Evaluation Metrics**: Detailed confusion matrices and classification reports for each model.
- **Real-Time Inference**: A function to predict the category of any Bangla text input.
---
## 📁 Dataset  
- **Source**: [Dataset Source](https://www.kaggle.com/datasets/sabbirhossainujjal/potrika-bangla-newspaper-datasets)
- **Language**: Bangla
- **Categories**: Includes domains such as Economy, Education, Entertainment, International, National, Science & Technology, Sports, and Politics.
- **Preprocessing Steps**:
  - Tokenized and padded sequences to a maximum length of 1000.
  - Labels converted to categorical format for model compatibility.
---
## 🏗️ Model Architectures

### GRU (Gated Recurrent Unit)
- **Embedding Layer**: Vector representation of words with a dimension of 1024.
- **GRU Layers**:
  - First GRU layer (128 units) with sequence return enabled.
  - Second GRU layer (64 units).
- **Dropout**: Regularization with a rate of 0.5.
- **Output Layer**: Dense layer with softmax activation for multi-class classification.

**Results**:
- **Accuracy**: Achieved high accuracy on the test set.
- **Insights**: GRU performed well with sequential dependencies in Bangla text.

### LSTM (Long Short-Term Memory)
- **Embedding Layer**: Same as GRU.
- **LSTM Layers**:
  - First LSTM layer (128 units) with sequence return enabled.
  - Second LSTM layer (64 units).
- **Dropout**: Same as GRU.
- **Output Layer**: Identical to GRU.

**Results**:
- **Accuracy**: Comparable to GRU but excelled in capturing long-term dependencies.
- **Insights**: LSTM's ability to retain information over extended sequences proved beneficial.

### CNN (Convolutional Neural Network)
- **Embedding Layer**: Same as GRU and LSTM.
- **Conv1D Layer**: 128 filters with a kernel size of 5.
- **GlobalMaxPooling**: Reduces dimensionality while preserving key features.
- **Dense Layers**:
  - Intermediate dense layer with 64 units and ReLU activation.
  - Final dense layer with softmax activation.

**Results**:
- **Accuracy**: Faster training and inference compared to RNN-based models.
- **Insights**: Effective in identifying localized patterns within the text.
---

## 📊 Performance Metrics

| Model | Accuracy |
|-------|----------|
| GRU   | 91.72%   |
| LSTM  | 90.99%   |
| CNN   | 90.60%   |

- **Confusion Matrix**: Visual representation for each model to assess misclassification trends.
- **Classification Report**: Includes precision, recall, and F1-scores for all categories.

---

## 🔧 Technologies Used  
- **Programming Language**: Python  
- **Frameworks & Libraries**: TensorFlow, Keras, NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn  
- **Tools**: Jupyter Notebook, Git  


## 🔮 Future Enhancements  
- Fine-tune models with transformer-based architectures (e.g., BERT).  
- Extend to multi-label classification for overlapping categories.  
- Deploy as a web application for user-friendly interaction.  

---
## 🤝 Acknowledgments

- Special thanks to the creators of the dataset.
- Inspired by the advancements in NLP for low-resource languages.

# 🌟 Happy Coding! 🌟

