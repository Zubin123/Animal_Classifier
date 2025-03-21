# Animal Image Classifier

This project is an image classification web application that predicts the animal species in an uploaded image using a Convolutional Neural Network (CNN) model trained on 15 animal classes.

## Features
- **Deep Learning Model**: Uses a CNN model built with TensorFlow/Keras.
- **Web Interface**: Flask-based web application for easy image upload and classification.
- **Local & Cloud Deployment**: Can be run locally or deployed on a cloud platform.

## Installation
### Prerequisites
- Python 3.9
- Anaconda (recommended)
- Virtual environment setup with required dependencies

### Setup Instructions
1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo/Animal_Classifier.git
   cd Animal_Classifier
   ```
2. Create and activate a virtual environment:
   ```sh
   conda create --name aenv python=3.9
   conda activate aenv
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. Run the application:
   ```sh
   python src/app.py
   ```

## Usage
- Open a web browser and go to `http://127.0.0.1:5000`
- Upload an image of an animal
- Get the predicted class along with confidence score

## Model Training
To retrain the model:
```sh
python src/train.py
```

## Deployment
### Local Deployment
Run the Flask application as shown above.

### Cloud Deployment
For deployment on platforms like AWS, Azure, or Heroku, follow the respective cloud service instructions and ensure Flask is configured to run on the appropriate host and port.

## Directory Structure
```
Animal_Classifier/
│── models/            # Saved CNN model
│── src/               # Source code
│   ├── train.py       # Model training script
│   ├── evaluate.py    # Model evaluation script
│   ├── app.py         # Flask web application
│── templates/         # HTML templates for Flask
│── static/            # CSS, JavaScript, images
│── uploads/           # Uploaded images
│── README.md          # Project documentation
│── requirements.txt   # Python dependencies
```

## Future Enhancements
- Improve UI/UX of the web application
- Add support for more animal classes
- Deploy to a cloud-based inference API

## License
MIT License

---
### Contributors
- Mohammed Zubin Essudeen

For any issues, please create a GitHub issue or reach out via email.

Happy Coding! 🚀
