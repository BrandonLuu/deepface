from typing import Any
import numpy as np
from deepface.models.Detector import Detector
from deepface.detectors import (
    OpenCvWrapper,
    SsdWrapper,
    DlibWrapper,
    MtcnnWrapper,
    RetinaFaceWrapper,
    MediapipeWrapper,
    YoloWrapper,
    YunetWrapper,
    FastMtcnnWrapper,
)


def build_model(detector_backend: str) -> Any:
    """
    Build a face detector model
    Args:
        detector_backend (str): backend detector name
    Returns:
        built detector (Any)
    """
    global face_detector_obj  # singleton design pattern

    backends = {
        "opencv": OpenCvWrapper.OpenCv,
        "mtcnn": MtcnnWrapper.MtCnn,
        "ssd": SsdWrapper.Ssd,
        "dlib": DlibWrapper.Dlib,
        "retinaface": RetinaFaceWrapper.RetinaFace,
        "mediapipe": MediapipeWrapper.MediaPipe,
        "yolov8": YoloWrapper.Yolo,
        "yunet": YunetWrapper.YuNet,
        "fastmtcnn": FastMtcnnWrapper.FastMtCnn,
    }

    if not "face_detector_obj" in globals():
        face_detector_obj = {}

    built_models = list(face_detector_obj.keys())
    if detector_backend not in built_models:
        face_detector = backends.get(detector_backend)

        if face_detector:
            face_detector = face_detector()
            face_detector_obj[detector_backend] = face_detector
        else:
            raise ValueError("invalid detector_backend passed - " + detector_backend)

    return face_detector_obj[detector_backend]


def detect_faces(detector_backend: str, img: np.ndarray, align: bool = True) -> list:
    """
    Detect face(s) from a given image
    Args:
        detector_backend (str): detector name
        img (np.ndarray): pre-loaded image
        alig (bool): enable or disable alignment after detection
    Returns
        result (list): tuple of face (np.ndarray), face region (list)
            , confidence score (float)
    """
    face_detector: Detector = build_model(detector_backend)
    return face_detector.detect_faces(img=img, align=align)