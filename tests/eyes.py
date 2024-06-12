# Extract the eyes from normalized face image

# 3rd party dependencies
import pickle
import os
import cv2
import numpy as np
import random
from pathlib import Path

# project dependencies
from deepface import DeepFace
from deepface.commons import logger as log

# Pickle testing flag to skip face analysis
PICKLE_TESTING_ENABLE = True
FLIP_RIGHT_EYE_X_Y = False # Changes R eye cropping coordinates - creates a jumping/flickering effect on video
TWO_PASS_EXTRACTION = False # Double extract for resolution loss(?) - some cropping from double normalization of face

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


def cv_show_wrapper(out_frame):
    if cv2.waitKey(0) == ord('q'): return True
    cv2.imshow('frame', out_frame)
    return False


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
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore
    
    # frame_width = 555 # normalized face image is (height=554, width=555)
    # frame_height = 554
    eye_crop_width = 50
    frame_width = eye_crop_width * 2
    frame_height = eye_crop_width * 2
    framerate = int(input_movie.get(cv2.CAP_PROP_FPS))

    # Create Output movies
    input_vid_filename = Path(input_vid).stem
    right_file = os.path.join(os.getcwd(), "face_only", f'{input_vid_filename}_right.mp4')
    left_file = os.path.join(os.getcwd(), "face_only", f'{input_vid_filename}_left.mp4')
    rand_file = os.path.join(os.getcwd(), "face_only", f'{input_vid_filename}_rand.mp4')
    out_movie_right = cv2.VideoWriter(right_file, fourcc, framerate, (frame_width, frame_height))
    out_movie_left = cv2.VideoWriter(left_file, fourcc, framerate, (frame_width, frame_height))
    out_movie_rand = cv2.VideoWriter(rand_file, fourcc, framerate, (frame_width, frame_height))

    logger.info(f"Output: R:{right_file} L:{left_file}")
    
    # Processing variables init
    frame_number = 0
    frame_face_objs = []
    eye_choice = "right"
    
    # Prefill Frame if Pickling
    if PICKLE_TESTING_ENABLE:  
        try:
            with open("face_objs.pkl", "rb") as in_pkl:
                logger.info("=== Found face_obj.pkl")
                frame_face_objs = pickle.load(in_pkl)
        except FileNotFoundError:
            logger.info("=== No pickle")
    
    # Processing Loop
    while input_movie.isOpened():

        # Grab a single frame of video
        ret, frame = input_movie.read()
        frame_number += 1
        face_obj = {}

        # Quit when the input video file ends
        if not ret:
            break
        
        if frame_number % 25 == 0:
            logger.info(f"Processing frame {frame_number} / {length}")

        # Extract Face per Frame - if pickling and file not created yet (for future testing)
        if PICKLE_TESTING_ENABLE and not os.path.isfile("face_objs.pkl"):
            # Note: pretty sure pickling is broken :shrug:
            # frame_face_objs.append(DeepFace.extract_faces(img_path=frame, detector_backend=detector_backends[0], align=True, expand_percentage=0))
            face_obj = DeepFace.extract_faces(img_path=frame, detector_backend=detector_backends[0], align=True, expand_percentage=0)
            face_obj = face_obj[0]
            frame_face_objs.append(face_obj["facial_area"])

        elif not os.path.isfile("face_objs.pkl"):
            # 1st Pass: Create normalized face
            face_obj = DeepFace.extract_faces(img_path=frame, detector_backend=detector_backends[0],enforce_detection=False, align=True, expand_percentage=0)
            face_obj = face_obj[0] # HARDCODE: processing only 1st detected face
            if TWO_PASS_EXTRACTION:
                face = convert_to_openCV_format(face_obj["face"]) # Convert to image for re-extraction
        
                # 2st Pass: process Eyes of normalized face with adjusted Eye coordinates
                face_obj = DeepFace.extract_faces(img_path=face, detector_backend=detector_backends[0], enforce_detection=False, align=True, expand_percentage=0)
                face_obj = face_obj[0]# HARDCODE: processing only 1st detected face
                face = convert_to_openCV_format(face_obj["face"])# Convert to image for outputting

        else:
            # logger.info("reading from pickle")
            face_obj["facial_area"] = frame_face_objs[frame_number-1]
            # cv_show_wrapper(frame_face_objs[frame_number]["right_eye"])
            
        # Get Eyes
        right_eye = face_obj["facial_area"]["right_eye"]
        left_eye = face_obj["facial_area"]["left_eye"]

        # Skip outputting if face detection failed on frame
        if right_eye is None or left_eye is None: continue

        if FLIP_RIGHT_EYE_X_Y:
            # Crop Eyes - switching right_eye [1] or [0] will give a jumping flickering output
            right_eye_crop = face[right_eye[1] - eye_crop_width : right_eye[1] + eye_crop_width,
                                right_eye[0] - eye_crop_width : right_eye[0] + eye_crop_width]
            left_eye_crop = face[left_eye[1] - eye_crop_width : left_eye[1] + eye_crop_width,
                                left_eye[0] - eye_crop_width : left_eye[0] + eye_crop_width]
        else:
            # Crop Eyes - Correctly indexed
            right_eye_crop = frame[right_eye[1] - eye_crop_width : right_eye[1] + eye_crop_width,
                                right_eye[0] - eye_crop_width : right_eye[0] + eye_crop_width]
            left_eye_crop = frame[left_eye[1] - eye_crop_width : left_eye[1] + eye_crop_width,
                                left_eye[0] - eye_crop_width : left_eye[0] + eye_crop_width]

        # Show each frame
        # frame = cv2.circle(frame, right_eye, 5, (0,0,255), 5 )
        # frame = cv2.circle(frame, left_eye, 5, (0,0,255), 5 )
        # comparison_img = np.hstack((cv2.resize(frame, (400,400)), cv2.resize(right_eye_crop, (400,400))))
        # cv2.imshow('frame', comparison_img)
        
        # cv_show_wrapper(right_eye_crop)

        # Write to outputs
        out_movie_right.write(right_eye_crop)
        out_movie_left.write(left_eye_crop)

        # # Random frame write
        rand_eye = right_eye_crop if eye_choice == "right" else left_eye_crop
        switch = random.randint(1, 2)
        if switch == 1:  # chance to flip the eye output
            if eye_choice == "right":
                # print("switching to left")
                rand_eye = left_eye_crop
                eye_choice = "left"
            else:
                # print("switching to right")
                rand_eye = right_eye_crop
                eye_choice = "right"

        out_movie_rand.write(rand_eye)


    # Write to pickle if no pickle created yet
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
    face_dir = os.path.join(os.getcwd(), 'face_only')
    input_file = os.path.join(face_dir, input_vid)
    logger.info("Extracting eyes of file: " + input_file)
    
    extract_and_output_eyes(input_file)
    
    cv2.destroyAllWindows()
