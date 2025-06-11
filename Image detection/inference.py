from flask import Flask, request, render_template, url_for, send_from_directory, redirect, flash
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing import image  # type: ignore
import numpy as np
import os
import subprocess
import json

class InferenceModel:
    """
    A class to load a trained model and handle file uploads for predictions.
    """

    def __init__(self, model_path):
        """
        Initialize the InferenceModel class.

        Args:
            model_path (str): Path to the saved Keras model.
        """
        self.model = load_model(model_path)
        self.app = Flask(__name__)
        self.app.secret_key = 'your_secret_key'  # Needed for flashing messages
        self.app.config['UPLOAD_FOLDER'] = 'uploads'
        os.makedirs(self.app.config['UPLOAD_FOLDER'], exist_ok=True)
        self.model_path = model_path

        @self.app.route('/', methods=['GET', 'POST'])
        def upload_file():
            """
            Handle file upload and prediction requests.

            Returns:
            --------
            str
                The rendered HTML template with the result or error message.
            """
            if request.method == 'POST':
                # Check if the post request has the file part
                if 'file' not in request.files:
                    return render_template('index.html', error='No file part')
                file = request.files['file']
                # If user does not select file, browser also
                # submits an empty part without filename
                if file.filename == '':
                    return render_template('index.html', error='No selected file')
                if file and self.allowed_file(file.filename):
                    # Save the uploaded file to the uploads directory
                    filename = file.filename
                    filepath = os.path.join(self.app.config['UPLOAD_FOLDER'], filename)
                    file.save(filepath)
                    # Extract EXIF data using exiftool
                    exif_data = self.extract_exif_data(filepath)
                    # Predict if the image is Real or Fake
                    label, confidence = self.predict_image(filepath)
                    # Prepare report data
                    report_data = {
                        'result': label,
                        'confidence': confidence,
                        'exif_data': exif_data,
                        'filename': filename
                    }
                    # Pass the report data to the template
                    return render_template('index.html', result=label, confidence=confidence, exif_data=exif_data, report_data=report_data)
                else:
                    return render_template('index.html', error='Allowed file types are png, jpg, jpeg')
            return render_template('index.html')

        @self.app.route('/report', methods=['POST'])
        def report():
            """
            Handle the report button actions.
            """
            action = request.form.get('action')
            if action == 'report_cybercrime':
                # Redirect to Cybercrime portal
                return redirect('https://cybercrime.gov.in/')
            elif action == 'wrong_results':
                # Flash a thank you message
                flash('Thank you for the feedback.')
                # Redirect back to home
                return redirect(url_for('upload_file'))
            else:
                # Invalid action
                flash('Invalid action.')
                return redirect(url_for('upload_file'))

        @self.app.route('/download_report', methods=['POST'])
        def download_report():
            """
            Handle the download report action.
            """
            # Retrieve report data from form
            report_data_json = request.form.get('report_data')
            if report_data_json:
                # Convert the report data back from JSON
                report_data = json.loads(report_data_json)
                # Render the report template
                rendered = render_template('report.html', **report_data)
                # Create a response with the rendered HTML
                response = self.app.make_response(rendered)
                response.headers['Content-Type'] = 'text/html; charset=utf-8'
                response.headers['Content-Disposition'] = f'attachment; filename=report_{report_data["filename"]}.html'
                return response
            else:
                flash('No report data available.')
                return redirect(url_for('upload_file'))

    def allowed_file(self, filename):
        """
        Check if a file has an allowed extension.

        Parameters:
        -----------
        filename : str
            The name of the file to check.

        Returns:
        --------
        bool
            True if the file has an allowed extension, False otherwise.
        """
        ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    def predict_image(self, file_path):
        """
        Predict whether an image is Real or Fake using the loaded model.

        Parameters:
        -----------
        file_path : str
            The path to the image file.

        Returns:
        --------
        tuple
            A tuple containing the prediction label ('Real' or 'Fake') and the confidence score.
        """
        # Load and preprocess image
        img = image.load_img(file_path, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array_expanded = np.expand_dims(img_array, axis=0)

        # Prediction logic
        result = self.model.predict(img_array_expanded)
        prediction = result[0][0]

        # Assuming the model outputs probabilities using sigmoid activation
        confidence = prediction * 100  # Confidence for 'Fake' class
        if prediction >= 0.5:
            label = 'Fake'
        else:
            label = 'Real'
            confidence = (1 - prediction) * 100  # Confidence for 'Real' class

        return label, confidence

    def extract_exif_data(self, file_path):
        """
        Extract EXIF data from an image file using exiftool.

        Parameters:
        -----------
        file_path : str
            The path to the image file.

        Returns:
        --------
        dict
            A dictionary containing EXIF data.
        """
        try:
            # Use exiftool to extract metadata
            cmd = ['exiftool', '-j', file_path]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode == 0:
                exif_json = json.loads(result.stdout)
                if exif_json:
                    exif_data = exif_json[0]
                    # Remove the 'SourceFile' entry
                    exif_data.pop('SourceFile', None)
                    return exif_data
                else:
                    return {'No EXIF data found': ''}
            else:
                return {'Error reading EXIF data': result.stderr}
        except Exception as e:
            return {'Error': str(e)}

    def run(self):
        """
        Run the Flask application with the loaded model.
        """
        self.app.run(debug=True)

if __name__ == '__main__':
    # Path to your saved Keras model
    model_path = 'deepfake_detector_model.keras'
    inference_model = InferenceModel(model_path)
    inference_model.run()
