# Extract the eyes from normalized face image

# 3rd party dependencies
import os
import cv2
import numpy as np
import random

# project dependencies
from deepface.commons import logger as log

logger = log.get_singletonish_logger()

def convert_to_openCV_format(img):
    # Scale float64(0-1) to uint8(0-255), then convert RGB image to BGR for openCV
    img = 255 * img
    img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def edit_eyes(in_right_file, in_left_file):
    # Open the input movie file
    in_movie_right = cv2.VideoCapture(in_right_file)
    in_movie_left = cv2.VideoCapture(in_left_file)
    length_right = int(in_movie_right.get(cv2.CAP_PROP_FRAME_COUNT))
    length_left = int(in_movie_left.get(cv2.CAP_PROP_FRAME_COUNT))

    # Configure output movie file (make sure resolution/frame rate matches input video!)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # type: ignore
    frame_width = 100 # TODO: this has to be configurable to match the frame compositor?
    frame_height = 100
    framerate = int(in_movie_right.get(cv2.CAP_PROP_FPS)) # TODO: how to output/respeed in/out framerate

    # Create Output movies
    out_file = os.path.join(os.getcwd(), "face_only", "right-left-edit","edit.avi")
    out_movie = cv2.VideoWriter(out_file, fourcc, framerate, (frame_width, frame_height))

    logger.info(out_file)
    
    # Processing variables init
    frame_number = 0
    eye_choice = "right"

    # Processing Loop
    while frame_number<350 and (in_movie_right.isOpened() or in_movie_left.isOpened()):

        # Grab a single frame of video
        frame_number += 1
        frame = {}
        ret_right, frame["right"] = in_movie_right.read()
        ret_left, frame["left"] = in_movie_left.read()
        
        print(f"Processing frame {frame_number} / R:{length_right} | L:{length_left} ")
        
        # Quit when the shortest input video file ends
        if ret_right is False or ret_left is False:
            break

        # Random frame write
        switch = random.randint(1, 4)
        if switch == 1:  # flip to other eye if success
            eye_choice = "left" if eye_choice == "right" else "right"
            print(f"switching to {eye_choice}")
            
        # Show each frame
        # if cv2.waitKey(10) == ord('q'): break
        # cv2.imshow('frame', frame[eye_choice])
        out_movie.write(frame[eye_choice])

    # Clean up
    in_movie_right.release()
    in_movie_left.release()
    out_movie.release()


if __name__ == "__main__":
    print("=== Beginning Edits ===")
    
    # File names
    right_vid = "right.avi"
    left_vid = "left.avi"
    
    # Get file from the test dir
    face_dir = os.path.join(os.getcwd(), "face_only", "right-left-edit")
    right_file = os.path.join(face_dir, right_vid)
    left_file = os.path.join(face_dir, left_vid)
    
    print(f"Editing Files: \nR: {right_file} \nL: {left_file}")
    
    edit_eyes(right_file, left_file)
    
    cv2.destroyAllWindows()
