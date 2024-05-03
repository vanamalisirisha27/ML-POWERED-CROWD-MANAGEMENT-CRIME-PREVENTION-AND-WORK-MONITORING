from pathlib import Path
import sys

# Get the absolute path of the current file
file_path = Path(__file__).resolve()

# Get the parent directory of the current file
root_path = file_path.parent

# Add the root path to the sys.path list if it is not already there
if root_path not in sys.path:
    sys.path.append(str(root_path))

# Get the relative path of the root directory with respect to the current working directory
ROOT = root_path.relative_to(Path.cwd())

# Sources
IMAGE = 'Image'
VIDEO = 'Video'
WEBCAM = 'Webcam'

SOURCES_LIST = [IMAGE, VIDEO, WEBCAM]

# Images config
IMAGES_DIR = ROOT / 'images'
CROWD_IMAGE = IMAGES_DIR / 'crowd.jpg'
CROWD_DETECT_IMAGE = IMAGES_DIR / 'crowd_detected.jpg'
VIOLENCE_IMAGE = IMAGES_DIR / 'violence.jpg'
VIOLENCE_DETECT_IMAGE = IMAGES_DIR / 'violence_detected.jpg'
WEAPON_IMAGE = IMAGES_DIR / 'weapon.jpg'
WEAPON_DETECT_IMAGE = IMAGES_DIR / 'weapon_detected.jpg'
WORK_IMAGE = IMAGES_DIR / 'work.jpg'
WORK_DETECT_IMAGE = IMAGES_DIR / 'work.jpg'
CRIME_IMAGE = IMAGES_DIR / 'criminal.jpg'
CRIME_DETECT_IMAGE = IMAGES_DIR / 'criminal.jpg'


# ML Model config
MODEL_DIR = ROOT / 'weights'
DETECTION_MODEL1 = MODEL_DIR / 'bestcrowd.pt'
DETECTION_MODEL2 = MODEL_DIR / 'best_violence.pt'
DETECTION_MODEL3 = MODEL_DIR / 'best_pistolsknives.pt'
DETECTION_MODEL4 = MODEL_DIR / 'best_work_monitoring.pt'
DETECTION_MODEL5 = MODEL_DIR / 'criminal_face_detection_resnetmodel.pth'
# In case of your custome model comment out the line above and
# Place your custom model pt file name at the line below 
# DETECTION_MODEL = MODEL_DIR / 'my_detection_model.pt'

SEGMENTATION_MODEL = MODEL_DIR / 'yolov8n-seg.pt'

# Webcam
WEBCAM_PATH = 0
