# Gesture Recognition System

A real-time hand gesture recognition system using Python, OpenCV, and MediaPipe. This system can recognize 31 different hand gestures and provide audio feedback.

## Features

- Real-time hand gesture recognition
- 31 different gesture recognition including:
  - Hello
  - No
  - Fist
  - Point
  - Peace
  - Thumbs Up
  - And many more...
- Audio feedback for recognized gestures
- Web interface for easy interaction
- Performance metrics (FPS, latency)

## Requirements

- Python 3.7+
- OpenCV
- MediaPipe
- Flask
- gTTS (Google Text-to-Speech)
- NumPy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/gesture-recognition.git
cd gesture-recognition
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

## Usage

1. Start the Flask server:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Allow camera access when prompted

4. Start making gestures in front of your camera

## Project Structure

```
gesture-recognition/
├── app.py              # Main application file
├── static/
│   ├── css/
│   │   └── style.css   # Stylesheet
│   └── audio/          # Generated audio files
├── templates/
│   └── index.html      # Web interface
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details. 