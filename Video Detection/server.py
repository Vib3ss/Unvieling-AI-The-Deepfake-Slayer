from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

# Interaction with the OS
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Used for DL applications, computer vision related processes
import torch
import torchvision

# For image preprocessing
from torchvision import transforms

# Dataset handling
from torch.utils.data.dataset import Dataset

import numpy as np
import cv2

# To recognize face from extracted frames
import face_recognition

# 'nn' helps us in creating & training neural networks
from torch import nn

# Contains definition for models for addressing different tasks
from torchvision import models

import warnings
warnings.filterwarnings("ignore")

# Import subprocess for running FFmpeg and yt-dlp
import subprocess
import uuid  # For generating unique filenames
import re    # For regex to detect platform

UPLOAD_FOLDER = 'Uploaded_Files'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app = Flask("__main__", template_folder="templates")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Creating Model Architecture
class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()

        # Returns a model pretrained on ImageNet dataset
        model = models.resnext50_32x4d(pretrained=True)

        # Sequential allows us to compose nn modules together
        self.model = nn.Sequential(*list(model.children())[:-2])

        # RNN to process input sequence
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)

        # Activation function
        self.relu = nn.LeakyReLU()

        # Dropout to avoid overfitting
        self.dp = nn.Dropout(0.4)

        # Fully connected layer
        self.linear1 = nn.Linear(2048, num_classes)

        # Adaptive average pooling
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape

        # Reshape input
        x = x.view(batch_size * seq_length, c, h, w)

        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))

im_size = 112

# Mean and std for normalization
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Softmax activation
sm = nn.Softmax(dim=1)

# For prediction
def predict(model, img):
    with torch.no_grad():
        fmap, logits = model(img)
        logits = sm(logits)
        _, prediction = torch.max(logits, 1)
        confidence = logits[:, int(prediction.item())].item() * 100
        print('Confidence of prediction: ', confidence)
    return [int(prediction.item()), confidence]

# Dataset class for per-frame analysis
class validation_dataset(Dataset):
    def __init__(self, video_names, transform=None):
        self.video_names = video_names
        self.transform = transform

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_path = self.video_names[idx]
        frames = []
        for frame in self.frame_extract(video_path):
            # Face detection and extraction
            faces = face_recognition.face_locations(frame)
            try:
                top, right, bottom, left = faces[0]
                frame = frame[top:bottom, left:right, :]
            except IndexError:
                pass  # If no face detected, use the original frame
            frames.append(self.transform(frame))
        return frames  # Return list of frames

    def frame_extract(self, path):
        vidObj = cv2.VideoCapture(path)
        success, image = vidObj.read()
        while success:
            yield image
            success, image = vidObj.read()

def detectFakeVideo(videoPath):
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((im_size, im_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    path_to_videos = [videoPath]

    video_dataset = validation_dataset(path_to_videos, transform=train_transforms)
    model = Model(2)
    path_to_model = 'model/df_model.pt'
    model.load_state_dict(torch.load(path_to_model, map_location=torch.device('cpu')))
    model.eval()
    frames = video_dataset[0]  # Get frames from the first (only) video
    predictions = []
    for idx, frame in enumerate(frames):
        frame = frame.unsqueeze(0).unsqueeze(0)  # Adjust dimensions to [batch_size, seq_length, c, h, w]
        prediction = predict(model, frame)
        predictions.append({
            'frame': idx,
            'output': 'REAL' if prediction[0] == 1 else 'FAKE',
            'confidence': prediction[1]
        })
    
    # Calculate overall prediction
    fake_count = sum(1 for pred in predictions if pred['output'] == 'FAKE')
    real_count = sum(1 for pred in predictions if pred['output'] == 'REAL')
    overall_output = 'FAKE' if fake_count > real_count else 'REAL'

    # Calculate average confidence
    avg_confidence = sum(pred['confidence'] for pred in predictions) / len(predictions)

    # Return both frame results and overall prediction
    return {
        'frame_results': predictions,
        'overall_output': overall_output,
        'average_confidence': avg_confidence
    }

@app.route('/', methods=['GET'])
def homepage():
    return render_template('index.html')

@app.route('/Detect', methods=['POST'])
def DetectPage():
    # Check if a file was uploaded
    if 'video' in request.files and request.files['video'].filename != '':
        video = request.files['video']
        print(video.filename)
        video_filename = secure_filename(video.filename)
        video.save(os.path.join(app.config['UPLOAD_FOLDER'], video_filename))
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)

        # Check the file extension
        file_ext = os.path.splitext(video_filename)[1].lower()
        if file_ext != '.mp4':
            # Convert to mp4
            output_filename = os.path.splitext(video_filename)[0] + ".mp4"
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
            # FFmpeg command to convert video
            command = [
                "ffmpeg",
                "-y",  # Overwrite output files without asking
                "-i", video_path,
                "-c:v", "libx264",
                "-c:a", "aac",
                "-strict", "experimental",
                "-b:a", "192k",
                "-b:v", "1000k",
                output_path
            ]
            try:
                # Run the command using subprocess
                subprocess.run(command, check=True)
                print(f"Conversion successful: {video_path} -> {output_path}")
                # Update video_path to point to the converted file
                video_path = output_path
                # Delete the original file
                os.remove(os.path.join(app.config['UPLOAD_FOLDER'], video_filename))
            except subprocess.CalledProcessError as e:
                print(f"Error occurred during conversion: {e}")
                # Handle the error, perhaps return an error message
                return jsonify({'error': 'An error occurred during video conversion.'}), 500

    elif 'video_url' in request.form and request.form['video_url'] != '':
        # Handle URL input
        video_url = request.form['video_url']
        # Generate a unique filename
        video_filename = str(uuid.uuid4()) + '.mp4'
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)

        # Determine the platform based on the URL
        if re.search(r'(youtube\.com|youtu\.be)', video_url):
            # YouTube URL
            yt_dlp_command = [
                "yt-dlp",
                "--no-check-certificate",
                "--merge-output-format", "mp4",
                "-o", video_path,
                video_url
            ]
        elif re.search(r'(twitter\.com|x\.com)', video_url):
            # Twitter URL
            yt_dlp_command = [
                "yt-dlp",
                video_url,
                "-o", video_path
            ]
        elif 'facebook.com' in video_url:
            # Facebook URL
            yt_dlp_command = [
                "yt-dlp",
                video_url,
                "-o", video_path
            ]
        else:
            return jsonify({'error': 'Unsupported URL or platform.'}), 400

        try:
            # Run the command using subprocess
            subprocess.run(yt_dlp_command, check=True)
            print(f"Video downloaded successfully: {video_url} -> {video_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error occurred during video download: {e}")
            # Handle the error, perhaps return an error message
            return jsonify({'error': 'An error occurred during video download.'}), 500
    else:
        return jsonify({'error': 'No video file or URL provided.'}), 400

    # Now proceed to process the video
    result = detectFakeVideo(video_path)
    data = {
        'frame_results': result['frame_results'],
        'overall_output': result['overall_output'],
        'average_confidence': result['average_confidence']
    }
    # Remove the video file after processing
    os.remove(video_path)
    return jsonify(data)

if __name__ == '__main__':
    app.run(port=3000)
