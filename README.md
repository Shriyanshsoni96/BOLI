# BOLI - Sign Language Translation

BOLI is an innovative sign language translation application that uses computer vision and machine learning to translate sign language gestures into spoken language in real-time.

## Features

- Real-time sign language gesture recognition
- Support for 30+ common sign language gestures
- Two-hand gesture detection
- Audio feedback for recognized gestures
- User-friendly interface with pistachio green theme
- Responsive design for all devices

## Technology Stack

- **Backend**: Python, Flask
- **Computer Vision**: OpenCV, MediaPipe
- **Frontend**: HTML, CSS, JavaScript
- **Audio**: gTTS (Google Text-to-Speech)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/boli.git
cd boli
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python app.py
```

5. Open your browser and navigate to:
```
http://localhost:5000
```

## Project Structure

```
boli/
├── app.py                 # Main Flask application
├── gesture_recognition.py # Gesture recognition logic
├── requirements.txt       # Project dependencies
├── Procfile              # Heroku deployment configuration
├── runtime.txt           # Python runtime version
├── static/               # Static files (CSS, JS, images)
│   ├── css/
│   └── audio/
├── templates/            # HTML templates
│   ├── index.html
│   ├── index2.html
│   ├── about.html
│   └── gestures.html
└── README.md            # Project documentation
```

## Usage

1. Start the application
2. Click "Start Camera" to begin gesture recognition
3. Perform sign language gestures in front of your camera
4. The application will recognize and display the gesture
5. Enable audio feedback to hear the recognized gestures

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Team

- Shriyansh Soni (Team Leader)
- Uday Rajput
- Gourav Jaiswal
- Adarsh Kalmodiya

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MediaPipe for hand tracking
- OpenCV for computer vision
- Flask for web framework
- Google Text-to-Speech for audio feedback 

