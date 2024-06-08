# Art edits of eyes

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


def overlay_image_random(out_image, top_image, bound_x, bound_y):
    overlay_image(out_image, top_image, 
                  random.randint(bound_x, bound_y), 
                  random.randint(bound_x, bound_y))


def overlay_image(out_image, top_image, start_x, start_y):
    if out_image is None or top_image is None: return
    
    # Overlay top_image onto out_image starting at offset x, y
    (out_len_x, out_len_y, out_len_z) = out_image.shape
    (top_len_x, top_len_y, top_len_z) = top_image.shape

    # Iterate through the top image
    for x in range(top_len_x):
        for y in range(top_len_y):
            for z in range(top_len_z):
                
                # Add the offsets to to get coordinates to overwrite for out_image
                out_x = x + start_x
                out_y = y + start_y
                
                # Bound checks the offset coordinates 
                if (0 <= out_x < out_len_x) and (0 <= out_y < out_len_y):
                    # print(x,y,z,out_image.shape,top_image.shape, out_x, out_y)
                    out_image[out_x, out_y, z] = top_image[x, y, z]


def edit_eyes(in_right_file, in_left_file):
    # Open the input movie file
    in_movie_right = cv2.VideoCapture(in_right_file)
    in_movie_left = cv2.VideoCapture(in_left_file)
    length_right = int(in_movie_right.get(cv2.CAP_PROP_FRAME_COUNT))
    length_left = int(in_movie_left.get(cv2.CAP_PROP_FRAME_COUNT))

    # Configure output movie file (make sure resolution/frame rate matches input video!)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # type: ignore
    frame_width = 200 # TODO: this has to be configurable to match the frame compositor?
    frame_height = 200
    framerate = int(in_movie_right.get(cv2.CAP_PROP_FPS)) # TODO: how to output/respeed in/out framerate

    # Create Output movies
    out_file = os.path.join(os.getcwd(), "face_only", "right-left-edit","edit.avi")
    out_movie = cv2.VideoWriter(out_file, fourcc, framerate, (frame_width, frame_height))

    logger.info(f"Outfile: {out_file}")
    
    # Processing variables init
    frame_number = 0
    eye_choice = "right"
    
    x_offset_right = random.randint(-75,275) # TODO: percentage of frame wd/ht
    y_offset_right = random.randint(-75,275)
    
    x_offset_left = random.randint(-75,275)
    y_offset_left = random.randint(-75,275)
    
    # Processing Loop
    while frame_number<350 and (in_movie_right.isOpened() or in_movie_left.isOpened()):
        # Init blank frame for writing
        out_frame = np.zeros((frame_width, frame_height, 3), dtype=np.uint8)

        # Read a single frame of video
        frame_number += 1
        in_frame = {}
        ret_right, in_frame["right"] = in_movie_right.read()
        ret_left, in_frame["left"] = in_movie_left.read()
        
        logger.info(f"Processing frame {frame_number} / R:{length_right} | L:{length_left} ")
        
        # Quit when the input video file ends
        if ret_right is False and ret_left is False:
            break

        # Random frame write
        if random.randint(1, 4) == 1:  # flip to other eye if success and other eye has frames
            # if eye_choice == "right":
            #     eye_choice = "left"
            # elif eye_choice == "left":
            #     eye_choice = "right"
            
            eye_choice = "right" if eye_choice == "left" else "left"
            # logger.info(f"Eye selected: {eye_choice}")
        
        # Check if current eye_choice is valid
        if ret_right == False and eye_choice == "right":
            eye_choice = "left"
        elif ret_left == False and eye_choice =="left":
            eye_choice = "right"
        
        # Random offset changes
        if random.randint(1, 4) == 1:
            x_offset_right = random.randint(-75,275)
            y_offset_right = random.randint(-75,275)
            # logger.info(f"R Moving {x_offset_right}, {y_offset_right}")
        if random.randint(1, 4) == 1:
            x_offset_left = random.randint(-75,275)
            y_offset_left = random.randint(-75,275)
            # logger.info(f"L Moving {x_offset_left}, {y_offset_left}")
        
        bg_select = random.randint(1,4)
        if bg_select == 1:
            # Whacky frame tile code
            resize_frame = np.resize(in_frame[eye_choice], (200,200,3))
            overlay_image(out_frame, resize_frame, 0, 0)
        elif bg_select == 2:
            resize_frame = np.resize(in_frame[eye_choice], (150,150,3))
            overlay_image(out_frame, resize_frame, 0, 0)
        else:
            # Stretch 2x to match output
            out_frame = cv2.resize(in_frame[eye_choice], None, fx = 2, fy = 2, interpolation = cv2.INTER_CUBIC)
        
        if ret_right:
            overlay_image(out_frame, in_frame["right"], x_offset_right, y_offset_right)
        if ret_left:
            overlay_image(out_frame, in_frame["left"], x_offset_left, y_offset_left)
        
        # Show each frame
        # if cv2.waitKey(0) == ord('q'): break
        # cv2.imshow('frame', out_frame)
        
        # cv2.imshow('frame', in_frame[eye_choice])

        out_movie.write(out_frame)

    # Clean up
    in_movie_right.release()
    in_movie_left.release()
    out_movie.release()


if __name__ == "__main__":
    logger.info("=== Beginning Edits ===")
    
    # File names
    right_vid = "right.avi"
    left_vid = "left.avi"
    
    # Get file from the test dir
    face_dir = os.path.join(os.getcwd(), "face_only", "right-left-edit")
    right_file = os.path.join(face_dir, right_vid)
    left_file = os.path.join(face_dir, left_vid)
    
    logger.info(f"Editing Files: \nR: {right_file} \nL: {left_file}")
    
    edit_eyes(right_file, left_file)
    
    cv2.destroyAllWindows()
