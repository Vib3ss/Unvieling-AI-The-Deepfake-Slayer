import os
from flask import Flask, request, render_template, send_file
import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib
from pydub import AudioSegment
import base64

app = Flask(__name__)

def allowed_file(filename):
    allowed_extensions = {"wav", "mp3", "aac", "flac"}
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions

def extract_mfcc_features(audio_path, n_mfcc=13, n_fft=2048, hop_length=512):
    try:
        audio_data, sr = librosa.load(audio_path, sr=None)
    except Exception as e:
        print(f"Error loading audio file {audio_path}: {e}")
        return None

    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    return np.mean(mfccs.T, axis=0)

def generate_waveform(audio_path, output_image_path):
    audio_data, sr = librosa.load(audio_path)
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(audio_data, sr=sr)
    plt.title('Waveform')
    plt.savefig(output_image_path)
    plt.close()

def generate_spectrogram(audio_path, output_image_path):
    audio_data, sr = librosa.load(audio_path)
    X = librosa.stft(audio_data)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar()
    plt.title('Spectrogram')
    plt.savefig(output_image_path)
    plt.close()

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:image/png;base64,{encoded_string}"

def analyze_audio(input_audio_path):
    model_filename = "svm_model.pkl"
    scaler_filename = "scaler.pkl"

    if not os.path.exists(input_audio_path):
        return "Error: The specified file does not exist."
    elif not input_audio_path.lower().endswith(".wav"):
        return "Error: The specified file is not a .wav file."

    mfcc_features = extract_mfcc_features(input_audio_path)
    if mfcc_features is not None:
        scaler = joblib.load(scaler_filename)
        mfcc_features_scaled = scaler.transform(mfcc_features.reshape(1, -1))

        svm_classifier = joblib.load(model_filename)
        prediction = svm_classifier.predict(mfcc_features_scaled)

        if prediction[0] == 0:
            return "The input audio is classified as genuine."
        else:
            return "The input audio is classified as deepfake."
    else:
        return "Error: Unable to process the input audio."

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "audio_file" not in request.files:
            return render_template("index.html", message="No file part")
        
        audio_file = request.files["audio_file"]
        if audio_file.filename == "":
            return render_template("index.html", message="No selected file")
        
        if audio_file and allowed_file(audio_file.filename):
            if not os.path.exists("uploads"):
                os.makedirs("uploads")
            
            try:
                audio_path = os.path.join("uploads", audio_file.filename)
                audio_file.save(audio_path)
                extension = os.path.splitext(audio_path)[1].lower()

                if extension != ".wav":
                    try:
                        # Convert to .wav format using pydub
                        sound = AudioSegment.from_file(audio_path, format=extension[1:])
                        wav_audio_path = os.path.splitext(audio_path)[0] + ".wav"
                        sound.export(wav_audio_path, format="wav")
                        os.remove(audio_path)  # Remove the original file
                        audio_path = wav_audio_path  # Update the path to point to the .wav file
                    except Exception as e:
                        os.remove(audio_path)
                        return render_template("index.html", message=f"Error converting file to .wav format: {e}")

                # Analyze the audio
                result = analyze_audio(audio_path)

                # Generate the waveform and spectrogram images
                waveform_image_path = os.path.join("uploads", "waveform.png")
                spectrogram_image_path = os.path.join("uploads", "spectrogram.png")
                generate_waveform(audio_path, waveform_image_path)
                generate_spectrogram(audio_path, spectrogram_image_path)

                # Convert images to Base64
                waveform_base64 = image_to_base64(waveform_image_path)
                spectrogram_base64 = image_to_base64(spectrogram_image_path)

                return render_template("result.html", 
                                       result=result, 
                                       waveform_image=waveform_base64, 
                                       spectrogram_image=spectrogram_base64)
            finally:
                # Ensure the uploaded and converted files are removed after processing
                if 'audio_path' in locals() and os.path.exists(audio_path):
                    os.remove(audio_path)
        
        return render_template("index.html", message="Invalid file format. Only .wav, .mp3, .aac, and .flac files are allowed.")
    
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=False)
