## American Sign Language Detector

## Features

- Real-time ASL gesture recognition.
- Machine learning-based classification.
- fastAPI web application interface.
- Modular, clean structure for experimentation.

## Tech Stack

- Python
- **scikit-learn** – model training and evaluation  
- **pandas** – data handling and preprocessing  
- **mediapipe** – real-time hand keypoint detection  
- **fastAPI + uvicorn** – API serving  

## Model

- Model files: `model.p`, `data.pickle`
- Supervised learning using keypoint data from hand landmarks.

## Web/API App

- `API/` contains a FastAPI backend.
- You can serve predictions using `uvicorn`.

## Installation & Running the App

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AyushAI14/AyushAI14-American_sign_language_detector.git
   cd AyushAI14-American_sign_language_detector
   python3 -m venv env
   source env/bin/activate
   pip install -r requirements.txt
   uvicorn API.app:app --reload
   Run the WebApp/index.html
   ```
## web app
![Image](https://github.com/user-attachments/assets/9d92dc55-934a-4d35-9fa1-8421c1950102)
