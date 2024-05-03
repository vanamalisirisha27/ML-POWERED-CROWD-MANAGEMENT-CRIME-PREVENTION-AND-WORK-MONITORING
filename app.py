# Python In-built packages
from pathlib import Path
import PIL
import supervision as sv
from pydub.playback import play
from playsound import playsound
# External packages
import streamlit as st
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

# Local Modules
import settings
import helper


def play_sound():
    sound_file = 'mixkit-classic-alarm-995.wav'  # Replace with the path to your sound file
    playsound(sound_file)

# Setting page layout
st.set_page_config(
    page_title="Crowd Management, Crime Prevention and  Work Monitoring",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Main page heading
st.title("Crowd Management, Crime Prevention and  Work Monitoring")

# Sidebar
st.sidebar.header("Model Config")

# Model Options
model_type = st.sidebar.radio(
    "Select Task", ['Crowd Management', 'Violence Detection','Weapon Detection','Work Monitoring','Criminals Face Recognition'])

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 10, 100, 40)) / 100

# Selecting Detection Or Segmentation
if model_type == 'Crowd Management':
    model_path = Path(settings.DETECTION_MODEL1)
    st.title('Crowd Management')
    default_image_path = str(settings.CROWD_IMAGE)
    default_detected_image_path = str(settings.CROWD_DETECT_IMAGE)
elif model_type == 'Violence Detection':
    model_path = Path(settings.DETECTION_MODEL2)
    st.title('Violence Detection')
    default_image_path = str(settings.VIOLENCE_IMAGE)
    default_detected_image_path = str(settings.VIOLENCE_DETECT_IMAGE)
elif model_type == 'Weapon Detection':
    model_path = Path(settings.DETECTION_MODEL3)
    st.title('Weapon Detection')
    default_image_path = str(settings.WEAPON_IMAGE)
    default_detected_image_path = str(settings.WEAPON_DETECT_IMAGE)
elif model_type == 'Work Monitoring':
    model_path = Path(settings.DETECTION_MODEL4)
    st.title('Work Monitoring')
    default_image_path = str(settings.WORK_IMAGE)
    default_detected_image_path = str(settings.WORK_DETECT_IMAGE)
elif model_type == 'Criminals Face Recognition':
    model_path = Path(settings.DETECTION_MODEL5)
    st.title('Criminals Face Recognition')
    default_image_path = str(settings.CRIME_IMAGE)
    default_detected_image_path = str(settings.CRIME_DETECT_IMAGE)


# Load Pre-trained ML Model
try:
    if model_type == 'Criminals Face Recognition':
        if model_path.suffix.lower() == '.pth':
            num_classes_custom = 10
            model = helper.ResNetRecognizer(num_classes_custom)
            model_custom_state_dict = torch.load(settings.DETECTION_MODEL5)
            model.load_state_dict(model_custom_state_dict)
            model.eval()
        else:
            model = helper.load_model(model_path)
    else:
        model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)


st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio(
    "Source", settings.SOURCES_LIST)

source_img = None
# If image is selected
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                #default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Default Image",
                         use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image",
                         use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img is None:
            #default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            st.image(default_detected_image_path, caption='Detected Image',
                     use_column_width=True)
        else:
            if st.sidebar.button('Detect'):
                if model_type == 'Criminals Face Recognition':
                    # Perform face recognition on the detected image
                    uploaded_image = uploaded_image.convert('RGB')
                    bounding_boxes, predicted_names = helper.recognize_criminal(uploaded_image, model)

                    # Draw bounding boxes and display the predicted criminal names on the image
                    annotated_image = helper.draw_boxes(uploaded_image, bounding_boxes, predicted_names)

                    # Display the annotated image
                    st.image(annotated_image, caption='Detected Image', use_column_width=True)
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
                    res = model.predict(uploaded_image,
                                        conf=confidence
                                        )
                    names = res[0].names
                    class_detections_values = []
                    for k, v in names.items():
                        class_detections_values.append(res[0].boxes.cls.tolist().count(k))
                    # create dictionary of objects detected per class
                    classes_detected = dict(zip(names.values(), class_detections_values))

                    boxes = res[0].boxes
                    res_plotted = res[0].plot()[:, :, ::-1]
                    st.image(res_plotted, caption='Detected Image',
                            use_column_width=True)
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
                                st.warning("Alert Sound Played 🚨")
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
                    try:
                        with st.expander("Detection Results"):
                            for box in boxes:
                                st.write(box.data)
                    except Exception as ex:
                        # st.write(ex)
                        st.write("No image is uploaded yet!")

elif source_radio == settings.VIDEO:
    helper.play_uploaded_video(confidence, model, model_type)
elif source_radio == settings.WEBCAM:
    helper.play_webcam(confidence, model, model_type)
else:
    st.error("Please select a valid source type!")