# from fastapi import FastAPI, UploadFile, File, WebSocket
# import pickle
# import cv2
# import mediapipe as mp
# import numpy as np
# import asyncio
# from typing import List

# from fastapi.middleware.cors import CORSMiddleware


# # Load the trained model
# model_dict = pickle.load(open('./model.p', 'rb'))
# model = model_dict['model']

# # Define labels
# labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K',
#                11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U',
#                21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

# # Initialize FastAPI app
# app = FastAPI()

# # Initialize MediaPipe Hands
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# hands_static = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
# hands_realtime = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

# # Maximum length based on your model
# max_length = 84

# # List to store predicted letters
# predicted_letters = []

# @app.get("/")
# def read_root():
#     return {"message": "Welcome to the Sign Language Recognition API"}

# # HTTP Endpoint for Multiple Image Upload Prediction
# @app.post("/predict/")
# async def predict_sign(files: List[UploadFile] = File(...)):
#     predictions = []  # Store predictions for each image

#     for file in files:
#         # Read the uploaded file
#         file_bytes = await file.read()
#         np_array = np.frombuffer(file_bytes, np.uint8)
#         frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

#         data_aux, x_, y_ = [], [], []
#         H, W, _ = frame.shape
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = hands_static.process(frame_rgb)

#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 for i in range(len(hand_landmarks.landmark)):
#                     x = hand_landmarks.landmark[i].x
#                     y = hand_landmarks.landmark[i].y
#                     x_.append(x)
#                     y_.append(y)

#                 for i in range(len(hand_landmarks.landmark)):
#                     x = hand_landmarks.landmark[i].x
#                     y = hand_landmarks.landmark[i].y
#                     data_aux.append(x - min(x_))
#                     data_aux.append(y - min(y_))

#             if len(data_aux) < max_length:
#                 data_aux.extend([0] * (max_length - len(data_aux)))
#             else:
#                 data_aux = data_aux[:max_length]

#             prediction = model.predict([np.asarray(data_aux)])
#             predicted_character = labels_dict[int(prediction[0])]

#             # Store the predicted character in the array
#             predicted_letters.append(predicted_character)
#             predictions.append(predicted_character)

#         else:
#             predictions.append("No hands detected")

#     return {"predictions": predictions}

# # WebSocket Endpoint for Real-time Camera Predictions
# @app.websocket("/ws/predict/")
# async def websocket_predict(websocket: WebSocket):
#     await websocket.accept()

#     # OpenCV Video Capture
#     cap = cv2.VideoCapture(0)
#     try:
#         while cap.isOpened():
#             data_aux, x_, y_ = [], [], []
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             H, W, _ = frame.shape
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = hands_realtime.process(frame_rgb)

#             if results.multi_hand_landmarks:
#                 for hand_landmarks in results.multi_hand_landmarks:
#                     for i in range(len(hand_landmarks.landmark)):
#                         x = hand_landmarks.landmark[i].x
#                         y = hand_landmarks.landmark[i].y
#                         x_.append(x)
#                         y_.append(y)

#                     for i in range(len(hand_landmarks.landmark)):
#                         x = hand_landmarks.landmark[i].x
#                         y = hand_landmarks.landmark[i].y
#                         data_aux.append(x - min(x_))
#                         data_aux.append(y - min(y_))

#                 if len(data_aux) < max_length:
#                     data_aux.extend([0] * (max_length - len(data_aux)))
#                 else:
#                     data_aux = data_aux[:max_length]

#                 prediction = model.predict([np.asarray(data_aux)])
#                 predicted_character = labels_dict[int(prediction[0])]

#                 # Store the predicted character in the array
#                 predicted_letters.append(predicted_character)

#                 # Draw landmarks and bounding box on frame
#                 for hand_landmarks in results.multi_hand_landmarks:
#                     mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#                 await websocket.send_text(f"Detected: {predicted_character}")
#             else:
#                 await websocket.send_text("No hands detected")

#             await asyncio.sleep(0.1)  # Control the speed of sending frames

#     except Exception as e:
#         print(f"Error: {e}")
#     finally:
#         cap.release()
#         await websocket.close()

# # Endpoint to retrieve stored predictions
# @app.get("/predictions/")
# def get_predictions():
#     return {"predicted_letters": predicted_letters}


# from fastapi.middleware.cors import CORSMiddleware

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allows all origins; limit to specific origins in production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )






from fastapi import FastAPI, UploadFile, File, WebSocket
import pickle
import cv2
import mediapipe as mp
import numpy as np
import asyncio

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Define labels
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K',
               11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U',
               21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

# Initialize FastAPI app
app = FastAPI()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands_static = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
hands_realtime = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

# Maximum length based on your model
max_length = 84

@app.get("/")
def read_root():
    return {"message": "Welcome to the Sign Language Recognition API"}


# HTTP Endpoint for Image Upload Prediction
@app.post("/predict/")
async def predict_sign(file: UploadFile = File(...)):
    # Read the uploaded file
    file_bytes = await file.read()
    np_array = np.frombuffer(file_bytes, np.uint8)
    frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    data_aux, x_, y_ = [], [], []
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_static.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        if len(data_aux) < max_length:
            data_aux.extend([0] * (max_length - len(data_aux)))
        else:
            data_aux = data_aux[:max_length]

        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]
        return {"prediction": predicted_character}

    return {"prediction": "No hands detected"}


# WebSocket Endpoint for Real-time Camera Predictions
@app.websocket("/ws/predict/")
async def websocket_predict(websocket: WebSocket):
    await websocket.accept()

    # OpenCV Video Capture
    cap = cv2.VideoCapture(0)
    try:
        while cap.isOpened():
            data_aux, x_, y_ = [], [], []
            ret, frame = cap.read()
            if not ret:
                break

            H, W, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands_realtime.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        x_.append(x)
                        y_.append(y)

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                if len(data_aux) < max_length:
                    data_aux.extend([0] * (max_length - len(data_aux)))
                else:
                    data_aux = data_aux[:max_length]

                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]

                # Draw landmarks and bounding box on frame
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                await websocket.send_text(f"Detected: {predicted_character}")
            else:
                await websocket.send_text("No hands detected")

            await asyncio.sleep(0.1)  # Control the speed of sending frames

    except Exception as e:
        print(f"Error: {e}")
    finally:
        cap.release()
        await websocket.close()
