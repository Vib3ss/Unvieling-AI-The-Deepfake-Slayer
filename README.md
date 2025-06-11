# Unveil AI - The Deepfake Slayer

## 🚀 Overview

This unified project combines three detection pipelines for identifying deepfakes across:

* **Videos** using CNN+LSTM (ResNeXt50)
* **Images** using TensorFlow/Keras-based CNN
* **Audio** using MFCC + SVM pipeline

Each module is independently usable and deployable, and offers either CLI or Flask-based web interface.

---

## 🎥 Video Deepfake Detection

### Features

* ResNeXt50 CNN as spatial encoder
* LSTM to capture temporal dependencies
* Face detection on each frame using `face_recognition`
* Automatic support for `.mp4` files and YouTube, Twitter, Facebook links

### Installation

```bash
pip install flask torch torchvision face_recognition opencv-python yt-dlp
sudo apt install ffmpeg  # or use brew on macOS
```

### Run Server

```bash
Download df_model.pt and make a folder with model name and put it in there.
python3 server.py
```

Then visit: `http://localhost:3000`

### File Structure

```
.
├── server.py
├── model/
│   └── df_model.pt
├── templates/
│   └── index.html
└── Uploaded_Files/ (auto-created)
```

---

## 📷 Image Deepfake Detection

### Features

* TensorFlow/Keras CNN with Rescaling, Conv2D, BatchNorm, MaxPooling
* Trained on Kaggle dataset: [Deepfake and Real Images](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images)

### Installation

```bash
pip install tensorflow keras flask numpy pillow
```

### Inference Web App

```bash
python inference.py
```

Then visit: `http://localhost:5000`

### Training (Optional)

```bash
python train.py  # downloads dataset and trains model
```

### Model Structure (Simplified)

```
Input -> Conv2D -> BatchNorm -> MaxPool -> Flatten -> Dense -> Dropout -> Sigmoid
```

### Dataset Citation

```
@Inproceedings{ltnghia-ICCV2021,
  Title = {OpenForensics: Large-Scale Challenging Dataset For Multi-Face Forgery Detection},
  Author = {Trung-Nghia Le et al.},
  BookTitle = {ICCV},
  Year = {2021},
}
```

---

## 🎧 Audio Deepfake Detection

### Features

* Uses MFCC features extracted from `.wav` files
* Trained using Support Vector Machine (SVM)

### Installation

```bash
pip install scikit-learn numpy librosa flask
```

### Run Web App

```bash
python app.py
```

### CLI Inference

```bash
python analyze_audio.py path/to/audio.wav
```

### Training

Organize your files:

```
real_audio/
    sample1.wav
    sample2.wav

deepfake_audio/
    df1.wav
    df2.wav
```

Run:

```bash
python main.py
```

## 🕹️ Future Improvements

* Unified dashboard UI for all three models
* Model conversion to ONNX for edge deployment
* RESTful API for all media types

---

> ✨ Ready to run deepfake detection on all three media types! Upload, analyze, and verify with confidence.
