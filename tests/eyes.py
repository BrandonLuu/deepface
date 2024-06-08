# Extract the eyes from normalized face image

# 3rd party dependencies
import pickle
import os
import cv2
import numpy as np
import random

# project dependencies
from deepface import DeepFace
from deepface.commons import logger as log

# Pickle testing flag to skip face analysis
PICKLE_TESTING_ENABLE = False

logger = log.get_singletonish_logger()

model_names = [
    # "VGG-Face",
    # "Facenet",
    # "Facenet512",
    # "OpenFace",
    # "DeepFace",
    # "DeepID",
    "Dlib",
    # "ArcFace",
    # "SFace",
    # "GhostFaceNet",
]

detector_backends = [
    # "opencv2",
    # "ssd",
    "dlib",
    # "mtcnn",
    # "fastmtcnn",
    # "mediapipe", # crashed in mac
    # "retinaface",
    # "yunet",
    # "yolov8",
    # "centerface",
]

# represent
# for model_name in model_names:
#     embedding_objs = DeepFace.represent(img_path="dataset/img1.jpg", model_name=model_name)
#     for embedding_obj in embedding_objs:
#         embedding = embedding_obj["embedding"]
#         logger.info(f"{model_name} produced {len(embedding)}D vector")


def convert_to_openCV_format(img):
    # Scale float64(0-1) to uint8(0-255), then convert RGB image to BGR for openCV
    img = 255 * img
    img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def extract_and_output_eyes(input_file):
    # Open the input movie file
    input_movie = cv2.VideoCapture(input_file)
    length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

    # Configure output movie file (make sure resolution/frame rate matches input video!)
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')  # type: ignore
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # type: ignore
    
    # frame_width = 555 # normalized face image is (height=554, width=555)
    # frame_height = 554
    eye_crop_width = 50
    frame_width = eye_crop_width * 2
    frame_height = eye_crop_width * 2
    framerate = int(input_movie.get(cv2.CAP_PROP_FPS))

    # Create Output movies
    right_file = os.path.join("tests", f'{input_vid}_right.mp4')
    left_file = os.path.join("tests", f'{input_vid}_left.mp4')
    rand_file = os.path.join("tests", f'{input_vid}_rand.mp4')
    out_movie_right = cv2.VideoWriter(right_file, fourcc, framerate, (frame_width, frame_height))
    out_movie_left = cv2.VideoWriter(left_file, fourcc, framerate, (frame_width, frame_height))
    out_movie_rand = cv2.VideoWriter(rand_file, fourcc, framerate, (frame_width, frame_height))

    if PICKLE_TESTING_ENABLE:  # prefill frame
        try:
            with open("face_objs.pkl", "rb") as in_pkl:
                logger.info("=== Found face_obj.pkl")
                frame_face_objs = pickle.load(in_pkl)
        except FileNotFoundError:
            logger.info("=== No pickle")

    # Processing variables init
    frame_number = 0
    frame_face_objs = []
    eye_choice = "right"
    
    # Processing Loop
    while (input_movie.isOpened()):

        # Grab a single frame of video
        ret, frame = input_movie.read()
        frame_number += 1

        # Quit when the input video file ends
        if not ret:
            break

        logger.info(f"Processing frame {frame_number} / {length}")

        # Extract Face per Frame - if pickling and file not created yet (for future testing)
        if PICKLE_TESTING_ENABLE and not os.path.isfile("face_objs.pkl"):
            # Note: pretty sure pickling is broken :shrug:
            frame_face_objs.append(DeepFace.extract_faces(img_path=frame, detector_backend=detector_backends[0], align=True, expand_percentage=0))
        else:
            # 1st Pass: Create normalized face
            face_obj = DeepFace.extract_faces(img_path=frame, detector_backend=detector_backends[0],enforce_detection=False, align=True, expand_percentage=0)
            face_obj = face_obj[0] # HARDCODE: processing only 1st detected face
            face = convert_to_openCV_format(face_obj["face"]) # Convert to image for re-extraction

            # 2st Pass: process Eyes of normalized face with adjusted Eye coordinates
            face_obj = DeepFace.extract_faces(img_path=face, detector_backend=detector_backends[0], enforce_detection=False, align=True, expand_percentage=0)
            face_obj = face_obj[0]# HARDCODE: processing only 1st detected face
            face = convert_to_openCV_format(face_obj["face"])# Convert to image for outputting

        # Get Eyes
        right_eye = face_obj["facial_area"]["right_eye"]
        left_eye = face_obj["facial_area"]["left_eye"]

        # Skip outputting if face detection failed on frame
        if right_eye is None or left_eye is None: continue

        # Crop Eyes
        right_eye_crop = face[right_eye[0] - eye_crop_width: right_eye[0] + eye_crop_width,
                              right_eye[1] - eye_crop_width: right_eye[1] + eye_crop_width]
        left_eye_crop = face[left_eye[1] - eye_crop_width: left_eye[1] + eye_crop_width,
                             left_eye[0] - eye_crop_width: left_eye[0] + eye_crop_width]

        # Show each frame
        # if cv2.waitKey(1000) == ord('q'): break
        # cv2.imshow('frame', face)
        # cv2.imshow('frame', frame)
        # cv2.imshow('frame', right_eye_crop)

        # Write to outputs
        out_movie_right.write(right_eye_crop)
        out_movie_left.write(left_eye_crop)

        # # Random frame write
        # rand_eye = right_eye_crop if eye_choice == "right" else left_eye_crop
        # switch = random.randint(1, 4)
        # if switch == 1:  # chance to flip the eye output
        #     if eye_choice == "right":
        #         print("switching to left")
        #         rand_eye = left_eye_crop
        #         eye_choice = "left"
        #     else:
        #         print("switching to right")
        #         rand_eye = right_eye_crop
        #         eye_choice = "right"

        # out_movie_rand.write(rand_eye)

    # Write to pickle
    if PICKLE_TESTING_ENABLE and not os.path.isfile("face_objs.pkl"):
        with open("face_objs.pkl", "wb") as out_pkl:
            pickle.dump(frame_face_objs, out_pkl, -1)

    # Clean up
    input_movie.release()
    out_movie_right.release()
    out_movie_left.release()


if __name__ == "__main__":
    logger.info("=== Beginning Eyes Processing ===")
    
    # Input file path
    input_vid = "girl_face_1920_1080_25fps.mp4"
    face_dir = os.path.join(os.getcwd(), "tests", 'face_only')
    input_file = os.path.join(face_dir, input_vid)
    logger.info("Extracting eyes of file: " + input_file)
    
    extract_and_output_eyes(input_file)
    
    cv2.destroyAllWindows()
