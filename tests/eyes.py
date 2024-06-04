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

def extract_faces(img_path, detector_backend, enable_pickle_testing=False):
    face_objs = None
    if enable_pickle_testing:
        try:
            with open("face_objs.pkl", "rb") as in_pkl:
                logger.info("=== Found face_obj.pkl")
                face_objs = pickle.load(in_pkl)
                
        except FileNotFoundError:
            logger.info("=== No face_obj.pkl --- mismatch flag/setup")
            
            # logger.info("=== No face_obj.pkl --- generating and saving")
            # face_objs = DeepFace.extract_faces(
            #     img_path=img_path,
            #     detector_backend=detector_backend,
            #     align=True,
            #     expand_percentage=0,
            # )
            # with open("face_objs.pkl", "wb") as out_pkl:
            #     pickle.dump(face_objs, out_pkl, -1)
            
    else:
        face_objs = DeepFace.extract_faces(
            img_path=img_path,
            detector_backend=detector_backend,
            align=True,
            expand_percentage=0,
        )
    
    return face_objs


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


def save_face(img_path, face_obj):
    # Hack to cleanup filenames and save to file
    # filename = (img_path.split("/")[1]).split(".jpg")[0]
    img_name = os.path.basename(img_path)
    
    plt.axis("off")
    plt.imshow(face_obj["face"])
    plt.savefig(f'face_only/face_{img_name}',  bbox_inches='tight')
    # plt.show()

    
def save_face_video(out_vid, face_obj):
    # Write the resulting image to the output video file
    out_vid.write(face_obj["face"])
    

def process_faces(img_path, out_vid, enable_save_faces=False):
        
    face_objs = extract_faces(img_path, detector_backends[0], enable_pickle_testing=True)
    
    for face_obj in face_objs:
        # face = face_obj["face"]
        # logger.info(f"testing {img_path} with {detector_backend}")
        # logger.info(f"testing with {detector_backend}")
        # logger.info(face_obj["facial_area"])
        # logger.info(face_obj["confidence"])
        
        """
        # # eye verification
        # # we know opencv2 sometimes cannot find eyes
        # if face_obj["facial_area"]["left_eye"] is not None:
        #     assert isinstance(face_obj["facial_area"]["left_eye"], tuple)
        #     assert isinstance(face_obj["facial_area"]["left_eye"][0], int)
        #     assert isinstance(face_obj["facial_area"]["left_eye"][1], int)

        # if face_obj["facial_area"]["right_eye"] is not None:
        #     assert isinstance(face_obj["facial_area"]["right_eye"], tuple)
        #     assert isinstance(face_obj["facial_area"]["right_eye"][0], int)
        #     assert isinstance(face_obj["facial_area"]["right_eye"][1], int)

        # # left eye is really the left eye of the person
        # if (
        #     face_obj["facial_area"]["left_eye"] is not None
        #     and face_obj["facial_area"]["right_eye"] is not None
        # ):
        #     re_x = face_obj["facial_area"]["right_eye"][0]
        #     le_x = face_obj["facial_area"]["left_eye"][0]
        #     assert re_x < le_x, "right eye must be the right eye of the person"

        # type_conf = type(face_obj["confidence"])
        # assert isinstance(
        #     face_obj["confidence"], float
        # ), f"confidence type must be float but it is {type_conf}"
        # assert face_obj["confidence"] <= 1
        """

        # ========== IMAGE MANIPULATION ==========
        if enable_save_faces:
            # save_face(img_path, face_obj)
            # save_face_video(out_vid, face_obj)
            # cv2.imshow("Frame", face_obj["face"])
            
            # plt.axis("off")
            # plt.imshow(face_obj["face"])
            # plt.show()
            # plt.savefig(f'face_only/face_test',  bbox_inches='tight')
            
            # out_vid.write(cv2.cvtColor(np.uint8(face_obj["face"]), cv2.COLOR_RGB2BGR))
            # cv2.imshow("frame", cv2.cvtColor(np.uint8(face_obj["face"]), cv2.COLOR_RGB2BGR))
            # cv2.imshow("Frame", cv2.cvtColor(face_obj["face"], cv2.COLOR_RGB2BGR))
            
            # x = (face_obj["face"]).shape
            # logger.info(f"shape: {x}")
            pass

        else:
            display_eyes(img_path, face_obj)


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
    frame_width = 555 # normalized face image is (height=554, width=555)
    frame_height = 554
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
                    
    
    while(input_movie.isOpened() and frame_number < 10 ):
        
        # Grab a single frame of video
        ret, frame = input_movie.read()
        frame_number += 1
        
        # Quit when the input video file ends
        if not ret:
            break

        logger.info(f"Processing frame {frame_number} / {length}")

        # Extract Face per Frame
        if PICKLE_TESTING_ENABLE and not os.path.isfile("face_objs.pkl"): # Extract if pickling and file not created yet (for future testing)
            frame_face_objs.append(DeepFace.extract_faces(img_path=frame, detector_backend=detector_backends[0], align=True, expand_percentage=0))
        else:
            face_obj = DeepFace.extract_faces(img_path=frame, detector_backend=detector_backends[0], align=True, expand_percentage=0)
            face = convert_to_openCV_format(face_obj[0]["face"])
        
        # if cv2.waitKey(1) == ord('q'):
        #     break
        # cv2.imshow('frame', face)
        # cv2.imshow('frame', frame)
        out_movie.write(face)
        

        
    # for frame_face_obj in frame_face_objs:
    #     face = frame_face_obj[0]["face"] # hardcoded [0] since each index is a detected face, but we're only doing single faces now
        
    #     # Scale float64(0-1) to uint8(0-255), then convert RGB image to BGR for openCV
    #     face = 255 * face
    #     face = face.astype(np.uint8)
    #     face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)

    #     # cv2.imshow('frame', face)
        
    #     # filename = f"face_only/frame_face{frame_number}.png"
    #     # cv2.imwrite(filename, face)
        
    #     if cv2.waitKey(1) == ord('q'):
    #         break
        
    #     # output_movie.write(face)
        
    if PICKLE_TESTING_ENABLE and not os.path.isfile("face_objs.pkl"): # Write out frame_face_objs
        with open("face_objs.pkl", "wb") as out_pkl:
            pickle.dump(frame_face_objs, out_pkl, -1)
        
    # Clean up
    input_movie.release()
    out_movie.release()
    cv2.destroyAllWindows()