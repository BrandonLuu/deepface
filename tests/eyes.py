# Extract the eyes from normalized face image

# 3rd party dependencies
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import pickle
import os

# project dependencies
from deepface import DeepFace
from deepface.commons import logger as log

# Pickle testing flag to skip face analysis
PICKLE_TESTING_ENABLE = False
SAVE_FACES_ENABLE = False

logger = log.get_singletonish_logger()

# some models (e.g. Dlib) and detectors (e.g. retinaface) do not have test cases
# because they require to install huge packages
# this module is for local runs

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
    # "opencv",
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
    if enable_pickle_testing:
        try:
            with open("face_objs.pkl", "rb") as in_pkl:
                logger.info("=== Found face_obj.pkl")
                face_objs = pickle.load(in_pkl)
                
        except FileNotFoundError:
            logger.info("=== No face_obj.pkl --- generating and saving")
            face_objs = DeepFace.extract_faces(
                img_path=img_path,
                detector_backend=detector_backend,
                align=True,
                expand_percentage=0,
            )
            with open("face_objs.pkl", "wb") as out_pkl:
                pickle.dump(face_objs, out_pkl, -1)
            
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

def process_faces(img_paths, enable_save_faces=False):
    expand_area = 0
    detector_backend = detector_backends[0]

    for img_path in img_paths:
        face_objs = extract_faces(img_path, detector_backend, enable_pickle_testing=PICKLE_TESTING_ENABLE)
        
        for face_obj in face_objs:
            # face = face_obj["face"]
            logger.info(f"testing {img_path} with {detector_backend}")
            logger.info(face_obj["facial_area"])
            logger.info(face_obj["confidence"])
            
            """
            # # eye verification
            # # we know opencv sometimes cannot find eyes
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
                save_face(img_path, face_obj)
            else:
                display_eyes(img_path, face_obj)

            logger.info("-----------")


if __name__ == "__main__":
    # img_paths = ["face_only/img11_face.png"]
    # img_paths = ["dataset/couple.jpg"]
    
    image = "img11.jpg"
    dataset_img_paths = ["dataset/"+image]
    face_only_img_paths = ["face_only/face_"+image]
    
    process_faces(dataset_img_paths, enable_save_faces=True)
    
    process_faces(face_only_img_paths, enable_save_faces=False)
    