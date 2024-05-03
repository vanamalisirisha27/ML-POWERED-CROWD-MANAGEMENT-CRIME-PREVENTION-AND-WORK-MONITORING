from ultralytics import YOLO
import time
import streamlit as st
import cv2
from pytube import YouTube
from playsound import playsound
import settings
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import torch
import torch.nn as nn
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision.models import resnet50, ResNet50_Weights
from retinaface import RetinaFace
import cv2

my_email = "vanamalisirisha27@gmail.com"
password_key = "puma tsde zxwz kumc"
gmail_server = "smtp.gmail.com"
gmail_port = 587

# Starting connection
my_server = smtplib.SMTP(gmail_server, gmail_port)
my_server.ehlo()
my_server.starttls()

# Login with your email and password
my_server.login(my_email, password_key)
# Create a MIMEMultipart message
message = MIMEMultipart("alternative")
text_content = "Alert:  Crowd Density is high"
message.attach(MIMEText(text_content))

recruiter_email = "20jg1a1254.sirisha@gvpcew.ac.in"

# Convert the MIMEMultipart object to a string
msg_string = message.as_string()


def play_sound():
    sound_file = 'mixkit-classic-alarm-995.wav'  
    playsound(sound_file)

def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    #model_path = str(model_path)
    model = YOLO(model_path)
    return model

# Define the ResNetRecognizer class for criminal recognition
class ResNetRecognizer(nn.Module):
    def __init__(self, num_classes):
        super(ResNetRecognizer, self).__init__()
        # Load pre-trained ResNet50 model
        self.resnet = resnet50(pretrained=True)
        # Freeze the ResNet50 layers
        for param in self.resnet.parameters():
            param.requires_grad = False
        # Replace the final fully connected layer for classification
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        # Pass input through ResNet50 model
        x = self.resnet(x)
        return x

# Define the mapping between class indices and criminal names
class_names = {
    0: "Andreas_yates",
    1: "Charles_Sobhraj",
    2: "Chhota_shakeel",
    3: "Jolly_Joseph",
    4: "Osama_bin_Laden",
    5: "Phoolan_devi",
    6: "Samantha",
    7: "Sandra_avila",
    8: "Veerappan",
    9: "Zakiur_rehman"
}
img_size=(224, 224)

def recognize_criminal(image, model):
    # Convert the PIL Image to a numpy array
    image_np = np.array(image)

    # Perform face detection with RetinaFace
    faces = RetinaFace.detect_faces(img_path=image_np)

    # Initialize lists to store bounding boxes and predicted criminal names
    bounding_boxes = []
    predicted_criminals = []

    # Check if faces is a list of dictionaries
    if isinstance(faces, dict):
        # Iterate over detected faces
        for face_key, face_info in faces.items():
            # Extract bounding box coordinates
            bbox = face_info["facial_area"]
            x1, y1, x2, y2 = bbox
            bounding_boxes.append(bbox)

            # Crop the face region from the original image
            face_region = image.crop((x1, y1, x2, y2))

            # Resize the face region to (224, 224) for model input
            resized_face = face_region.resize((224, 224))

            # Convert the image to numpy array
            resized_face_np = np.array(resized_face)

            # Convert BGR image to RGB
            resized_face_rgb = cv2.cvtColor(resized_face_np, cv2.COLOR_BGR2RGB)

            # Convert the image to PIL Image
            pil_image = Image.fromarray(resized_face_rgb)

            # Preprocess the image
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            preprocessed_image = transform(pil_image).unsqueeze(0)

            # Perform inference
            with torch.no_grad():
                outputs = model(preprocessed_image)
                _, predicted_index = torch.max(outputs, 1)

            # Map the predicted index to criminal names
            predicted_name = class_names.get(predicted_index.item(), "Unknown")
            predicted_criminals.append(predicted_name)

    return bounding_boxes, predicted_criminals


# Function to draw bounding boxes and text on an image
def draw_boxes(image, bounding_boxes, predicted_criminals):
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arial.ttf", 20)
    for bbox, criminal_name in zip(bounding_boxes, predicted_criminals):
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline="red")
        draw.text((x1, y1), f"{criminal_name}", fill="red",font=font)
    return image


def display_tracker_options():
    display_tracker = st.radio("Display Tracker", ('Yes', 'No'))
    is_display_tracker = True if display_tracker == 'Yes' else False
    if is_display_tracker:
        tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))
        return is_display_tracker, tracker_type
    return is_display_tracker, None


def _display_detected_frames(conf, model, model_type, st_frame, image, is_display_tracking=None, tracker=None):
    """
    Display the detected objects on a video frame using the YOLOv8 model.

    Args:
    - conf (float): Confidence threshold for object detection.
    - model (YoloV8): A YOLOv8 object detection model.
    - st_frame (Streamlit object): A Streamlit object to display the detected video.
    - image (numpy array): A numpy array representing the video frame.
    - is_display_tracking (bool): A flag indicating whether to display object tracking (default=None).

    Returns:
    None
    """
    print("Displaying detected frames...")
    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720*(9/16))))

    # Display object tracking, if specified
    if is_display_tracking:
        if model_type == 'Criminals Face Recognition':
            # Perform face recognition on the detected image
            uploaded_image = Image.fromarray(image).convert('RGB')
            bounding_boxes, predicted_names = recognize_criminal(uploaded_image, model)

            # Draw bounding boxes and display the predicted criminal names on the image
            annotated_image = draw_boxes(uploaded_image, bounding_boxes, predicted_names)

            # Convert the annotated image back to numpy array
            annotated_image_np = np.array(annotated_image)

            # Display the annotated image
            st_frame.image(annotated_image_np, caption='Detected Image', use_column_width=True)
            # Display the predicted criminal names
            for criminal_name in predicted_names:
                st.write(f"Predicted Criminal: {criminal_name}")
                play_sound()
                st.warning("Alert Sound Played 🚨")
                text_content = f"Alert:  Criminal {criminal_name} found"
                message.attach(MIMEText(text_content))
                msg_string = message.as_string()
                my_server.sendmail(from_addr=my_email, to_addrs=recruiter_email, msg=msg_string)
                st.success("Email Sent")
                my_server.quit()
        else:
            res = model.track(image, conf=conf, persist=True, tracker=tracker)
            names = res[0].names
            class_detections_values = []
            for k, v in names.items():
                class_detections_values.append(res[0].boxes.cls.tolist().count(k))
            # create dictionary of objects detected per class
            classes_detected = dict(zip(names.values(), class_detections_values))
            if 'people' in classes_detected:
                no_of_people = classes_detected['people']
                #st.title(f"No of People: {no_of_people}")
                if no_of_people:
                    if no_of_people == 0:
                        st.info("Crowd Density: Very Low")
                        st.success("No of people: {}".format(no_of_people))
                    elif 1< no_of_people < 10:
                        st.info("Crowd Density: Low")
                        st.success("No of people: {}".format(no_of_people))
                    elif 10< no_of_people <50:
                        st.info("Crowd Density: Medium")
                        st.success("No of people: {}".format(no_of_people))
                        play_sound()
                        st.warning("Alert Sound Played 🚨")
                    if no_of_people > 50:
                        st.title("Crowd Density: High")
                        st.title("No of people: {}".format(no_of_people))
                        play_sound()
                        my_server.sendmail(from_addr=my_email, to_addrs=recruiter_email, msg=msg_string)
                        st.success("Email Sent")
                        my_server.quit()
                else:
                    st.title("No People")
            if 'Violence' in classes_detected:
                if classes_detected['Violence'] > 0 :
                    st.info("Violence Detected")
                    play_sound()
                    st.warning("Alert Sound Played 🚨")
                    text_content = "Alert:  Violence Detected"
                    message.attach(MIMEText(text_content))
                    msg_string = message.as_string()
                    my_server.sendmail(from_addr=my_email, to_addrs=recruiter_email, msg=msg_string)
                    st.success("Email Sent")
                    my_server.quit()
            if 'NonViolence' in classes_detected:
                if classes_detected['NonViolence'] > 0 and classes_detected['Violence'] < 1:
                    st.info("No Violence Detected")
            if 'guns' in classes_detected:
                if classes_detected['guns'] > 0:
                    st.info("Gun Detected")
                    play_sound()
                    st.warning("Alert Sound Played 🚨")
                    text_content = "Alert:  Gun Detected"
                    message.attach(MIMEText(text_content))
                    msg_string = message.as_string()
                    my_server.sendmail(from_addr=my_email, to_addrs=recruiter_email, msg=msg_string)
                    st.success("Email Sent")
                    my_server.quit()
            if 'knife' in classes_detected:
                if classes_detected['knife'] > 0 :
                    st.info("Knife Detected")
                    play_sound()
                    st.warning("Alert Sound Played 🚨")
                    text_content = "Alert:  Knife Detected"
                    message.attach(MIMEText(text_content))
                    msg_string = message.as_string()
                    my_server.sendmail(from_addr=my_email, to_addrs=recruiter_email, msg=msg_string)
                    st.success("Email Sent")
                    my_server.quit()
            if 'Knife' in classes_detected:
                if classes_detected['Knife'] > 0 :
                    st.info("Knife Detected")
                    play_sound()
                    st.warning("Alert Sound Played 🚨")
                    text_content = "Alert:  Knife Detected"
                    message.attach(MIMEText(text_content))
                    msg_string = message.as_string()
                    my_server.sendmail(from_addr=my_email, to_addrs=recruiter_email, msg=msg_string)
                    st.success("Email Sent")
                    my_server.quit()
            if 'Pistol' in classes_detected:
                if classes_detected['Pistol'] > 0:
                    st.info("Pistol Detected")
                    play_sound()
                    st.warning("Alert Sound Played 🚨")
                    text_content = "Alert:  Pistol Detected"
                    message.attach(MIMEText(text_content))
                    msg_string = message.as_string()
                    my_server.sendmail(from_addr=my_email, to_addrs=recruiter_email, msg=msg_string)
                    st.success("Email Sent")
                    my_server.quit()
            if 'Desk' in classes_detected:
                if classes_detected['Desk'] > 0:
                    st.info("Desk")
            if 'Employee at desk' in classes_detected:
                if classes_detected['Employee at desk'] > 0:
                    st.info("Employee is at desk")
            if 'Empty chair' in classes_detected:
                if classes_detected['Empty chair'] > 0:
                    st.info("Empty chair")
                    play_sound()
                    st.warning("Alert Sound Played 🚨")
                    text_content = "Alert:  Empty Chair Detected"
                    message.attach(MIMEText(text_content))
                    msg_string = message.as_string()
                    my_server.sendmail(from_addr=my_email, to_addrs=recruiter_email, msg=msg_string)
                    st.success("Email Sent")
                    my_server.quit()
            if 'Moving Employee' in classes_detected:
                if classes_detected['Moving Employee'] > 0:
                    st.info("Moving Employee")
                    play_sound()
                    st.warning("Alert Sound Played 🚨")
                    text_content = "Alert:  Employee Moving Away"
                    message.attach(MIMEText(text_content))
                    msg_string = message.as_string()
                    my_server.sendmail(from_addr=my_email, to_addrs=recruiter_email, msg=msg_string)
                    st.success("Email Sent")
                    my_server.quit()
            if 'walk way' in classes_detected:
                if classes_detected['walk way'] > 0:
                    st.info("walk way")
                    play_sound()
                    st.warning("Alert Sound Played 🚨")
    else:
        # Predict the objects in the image using the YOLOv8 model
        res = model.predict(image, conf=conf)
        names = res[0].names
        class_detections_values = []
        for k, v in names.items():
            class_detections_values.append(res[0].boxes.cls.tolist().count(k))
        # create dictionary of objects detected per class
        classes_detected = dict(zip(names.values(), class_detections_values))

    # # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                caption='Detected Video',
                channels="BGR",
                use_column_width=True
                )


def play_webcam(conf, model,model_type):
    """
    Plays a webcam stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_webcam = settings.WEBCAM_PATH
    is_display_tracker, tracker = display_tracker_options()
    if st.sidebar.button('Detect Objects'):
        try:
            vid_cap = cv2.VideoCapture(source_webcam)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                with st.container():
                    success, image = vid_cap.read()
                    if success:
                        _display_detected_frames(conf,
                                                model,
                                                model_type,
                                                st_frame,
                                                image,
                                                is_display_tracker,
                                                tracker,
                                                )
                    else:
                        vid_cap.release()
                        break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))


def display_video_options():
    #video_option = st.radio("Video Source", ("Upload Video", "Choose from Stored Videos"))
    #video_option = st.radio("Video Source", "Upload Video")
    #if video_option == "Upload Video":
        uploaded_video = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
        return uploaded_video
'''
    else:
        selected_video = st.selectbox("Choose a stored video", list(settings.VIDEOS_DICT.keys()))
        return settings.VIDEOS_DICT[selected_video] '''

import os
import tempfile

def play_uploaded_video(conf, model, model_type):
    """
    Plays an uploaded video file. Tracks and detects objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    uploaded_video = display_video_options()
    is_display_tracker, tracker = display_tracker_options()  # Capture the values here

    if uploaded_video is not None:
        try:
            # Convert uploaded video to bytes object
            video_bytes = uploaded_video.read()

            # Save uploaded video to a temporary file
            with tempfile.NamedTemporaryFile(delete=False) as f:
                f.write(video_bytes)
                video_path = f.name

            # Create video capture object using the saved video file path
            vid_cap = cv2.VideoCapture(video_path)
            st_frame = st.empty()

            if st.sidebar.button('Detect Objects'):  # Add Detect Objects button
                while vid_cap.isOpened():
                    success, image = vid_cap.read()
                    if success:
                        _display_detected_frames(conf,
                                                 model,
                                                 model_type,
                                                 st_frame,
                                                 image,
                                                 is_display_tracker,  # Pass the values here
                                                 tracker
                                                 )
                    else:
                        vid_cap.release()
                        break

            # Release video capture object
            vid_cap.release()

            # Remove the temporary video file
            os.unlink(video_path)

        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))
    else:
        st.write("Upload a video file to detect objects.")
