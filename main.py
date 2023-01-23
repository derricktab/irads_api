from fastapi import FastAPI
from fastapi.responses import StreamingResponse
# import cv2
# import numpy as np
# from keras.models import load_model
import os
from twilio.rest import Client


# # Load the model
# model = load_model('model/model.h5')

# CAMERA can be 0 or 1 based on default camera of your computer.
videoFile = "accidents/MOST DANGEROUS ACCIDENTS IN UGANDA.mp4"
# camera = cv2.VideoCapture(videoFile)
app = FastAPI()

@app.get('/')
def main():
    return "THIS IS THE HOME PAGE OF THE API"


@app.get("detection")
def detection():
    def iterfile():
        with open(videoFile, mode="rb") as file_like:
            yield from file_like

    return StreamingResponse(iterfile(), media_type="video/mp4")

@app.post("/send-sms/{recipient}")
async def send_sms(recipient: str, message: str, auth_token: str):

    account_sid = "AC26fdfd8bdd076d809bb9ea04f0a9c42d"
    client = Client(account_sid, auth_token)

    message = client.messages.create(
        body=message,
        from_='+16692300626',
        to=recipient
    )
    return {"message": "SMS sent successfully!"}





# @app.get("/")
# def main():
#     def iterfile():
#         with open(videoFile, mode="rb") as file_like:
#             while True:
#                 # Grab the webcameras image.
#                 ret, original_image = camera.read()
#                 # Resize the raw image into (224-height,224-width) pixels.
#                 original_image = cv2.resize(
#                     original_image, (224, 224), interpolation=cv2.INTER_AREA)
#                 print(original_image.shape)
#                 image = original_image
#                 # Show the image in a window
#                 # Make the image a numpy array and reshape it to the models input shape.
#                 image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
#                 # Normalize the image array
#                 image = (image / 127.5) - 1
#                 # Have the model predict what the current image is. Model.predict
#                 # returns an array of percentages. Example:[0.2,0.8] meaning its 20% sure
#                 # it is the first label and 80% sure its the second label.
#                 probabilities = model.predict(image)
#                 predicted_class = np.argmax(probabilities)
#                 prediction = ""
#                 if (predicted_class == 0):
#                     prediction = "ACCIDENT"

#                 cv2.putText(original_image, prediction, (100, 30),
#                             cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 2)
#             yield from file_like

#     return StreamingResponse(iterfile(), media_type="video/mp4")

#     cv2.imshow('IRADS', original_image)

#     # else:
#     #     print("NO ACCIDENT")

#     # Listen to the keyboard for presses.
#     keyboard_input = cv2.waitKey(1)
#     # 27 is the ASCII for the esc key on your keyboard.
#     if keyboard_input == 27:
#         break

# camera.release()
# cv2.destroyAllWindows()
