# Extract the eyes from normalized face image

# 3rd party dependencies
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import pickle
import os
import cv2
import numpy as np

# project dependencies
from deepface import DeepFace
from deepface.commons import logger as log

# Pickle testing flag to skip face analysis
PICKLE_TESTING_ENABLE = False
SAVE_FACES_ENABLE = False

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


def display_eyes(img_path, face_obj):
        fig, ax = plt.subplots(1,3)
        face_img = plt.imread(img_path)
        right_eye = face_obj["facial_area"]["right_eye"]
        left_eye = face_obj["facial_area"]["left_eye"]

        # Plot Face
        ax[0].plot(right_eye[0], right_eye[1], "ro")
        ax[0].plot(left_eye[0], left_eye[1], "ro")
        ax[0].imshow(face_img)
        # plt.axis("off")

        # Crop Eyes
        eye_crop_width = 50
        right_eye_crop = face_img[ right_eye[0] - eye_crop_width : right_eye[0] + eye_crop_width, \
                                right_eye[1] - eye_crop_width : right_eye[1] + eye_crop_width]

        left_eye_crop =  face_img[ left_eye[1] - eye_crop_width : left_eye[1] + eye_crop_width, \
                                left_eye[0] - eye_crop_width : left_eye[0] + eye_crop_width]

        # Plot Eyes
        ax[1].imshow(right_eye_crop)
        ax[2].imshow(left_eye_crop)
        
        plt.show()


def convert_to_openCV_format(face):
    # Scale float64(0-1) to uint8(0-255), then convert RGB image to BGR for openCV
    face = 255 * face
    face = face.astype(np.uint8)
    face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
    return face


if __name__ == "__main__":
    
    # Open the input movie file
    input_movie = cv2.VideoCapture("face_only/girl_face_1920_1080_25fps.mp4")
    length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create an output movie file (make sure resolution/frame rate matches input video!)
    fourcc = cv2.VideoWriter_fourcc(*'XVID') # type: ignore
    # frame_width = 555 # normalized face image is (height=554, width=555)
    # frame_height = 554
    eye_crop_width = 50
    frame_width = eye_crop_width * 2
    frame_height = eye_crop_width * 2
    framerate = int(input_movie.get(cv2.CAP_PROP_FPS))
    
    out_movie = cv2.VideoWriter('face_only/output.avi', fourcc, framerate, (frame_width, frame_height))
    
    frame_number = 0
    frame_face_objs = []
    if PICKLE_TESTING_ENABLE: # prefill frame
        try:
            with open("face_objs.pkl", "rb") as in_pkl:
                logger.info("=== Found face_obj.pkl")
                frame_face_objs = pickle.load(in_pkl)
        except FileNotFoundError:
            logger.info("=== No pickle")
                    
    
    # while(input_movie.isOpened() and frame_number < 120 ):
    while(input_movie.isOpened()):
        
        # Grab a single frame of video
        ret, frame = input_movie.read()
        frame_number += 1
        
        # Quit when the input video file ends
        if not ret: break

        logger.info(f"Processing frame {frame_number} / {length}")

        # Extract Face per Frame
        if PICKLE_TESTING_ENABLE and not os.path.isfile("face_objs.pkl"): # Extract if pickling and file not created yet (for future testing)
            frame_face_objs.append(DeepFace.extract_faces(img_path=frame, detector_backend=detector_backends[0], align=True, expand_percentage=0))
        else:
            # 1st Pass: Create normalized face
            face_obj = DeepFace.extract_faces(img_path=frame, detector_backend=detector_backends[0], enforce_detection=False, align=True, expand_percentage=0)
            face_obj = face_obj[0] # HARDCODE: processing only 1st detected face
            face = convert_to_openCV_format(face_obj["face"])
            
            # 2st Pass: process Eyes of normalized face with adjusted Eye coordinates
            face_obj = DeepFace.extract_faces(img_path=face, detector_backend=detector_backends[0], enforce_detection=False, align=True, expand_percentage=0)
            face_obj = face_obj[0]
            face = convert_to_openCV_format(face_obj["face"])
            
        # Crop Eyes
    
        right_eye = face_obj["facial_area"]["right_eye"]
        left_eye = face_obj["facial_area"]["left_eye"]        
        if right_eye is not None:
            right_eye_crop = face[ right_eye[0] - eye_crop_width : right_eye[0] + eye_crop_width, \
                                    right_eye[1] - eye_crop_width : right_eye[1] + eye_crop_width]
        # if left_eye is not None:
            # left_eye_crop =  face[ left_eye[1] - eye_crop_width : left_eye[1] + eye_crop_width, \
            #                         left_eye[0] - eye_crop_width : left_eye[0] + eye_crop_width]
            
            # Show each frame
            # if cv2.waitKey(1000) == ord('q'): break
            # cv2.imshow('frame', face)
            # cv2.imshow('frame', frame)
            # cv2.imshow('frame', right_eye_crop)
            
            # Write to output
            out_movie.write(right_eye_crop)
        
    if PICKLE_TESTING_ENABLE and not os.path.isfile("face_objs.pkl"): # Write out frame_face_objs
        with open("face_objs.pkl", "wb") as out_pkl:
            pickle.dump(frame_face_objs, out_pkl, -1)
        
    # Clean up
    input_movie.release()
    out_movie.release()
    cv2.destroyAllWindows()
    