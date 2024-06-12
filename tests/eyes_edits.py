# Art edits of eyes

# 3rd party dependencies
import os
import cv2
import numpy as np
import random

# project dependencies
from deepface.commons import logger as log

logger = log.get_singletonish_logger()

# Output Dimensions
out_width = 200
out_height = 200


# === OpenCV Utility Functions ===
def convert_to_openCV_format(img):
    # Scale float64(0-1) to uint8(0-255), then convert RGB image to BGR for openCV
    img = 255 * img
    img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def cv_show_wrapper(out_frame):
    if cv2.waitKey(0) == ord('q'): return True
    cv2.imshow('frame', out_frame)
    return False

# === Editing Functions ===
def generate_grid(rows, cols, max_rand_val):
    grid = np.ndarray((rows,cols), dtype=int)
    row, col = grid.shape
    for i in range(row):
        for j in range(col):
            grid[i][j] = random.randint(0, max_rand_val)
    return grid


def print_grid(grid):
    for row in grid:
        print(row)


def overlay_image_random(out_image, top_image, bound_x, bound_y):
    overlay_image(out_image, top_image, random.randint(bound_x, bound_y),
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


def change_background(out_frame, in_frame):
    # Background select
    bg_select = random.randint(1,4)
    if bg_select == 1: # np resize full scale - tile
        # Whacky frame tile code
        resize_frame = np.resize(in_frame, (out_height, out_width, 3))
        resize_frame = cv2.resize(resize_frame, (out_height, out_width), interpolation = cv2.INTER_LINEAR)
        overlay_image(out_frame, resize_frame, 0, 0)
        
    elif bg_select == 2: # np resize 75% - stutter tile
        resize_percentage = 0.75
        resize_frame = np.resize(in_frame, (int(out_height * resize_percentage), int(out_width * resize_percentage), 3))
        resize_frame = cv2.resize(resize_frame, (out_height, out_width), interpolation = cv2.INTER_LINEAR)
        overlay_image(out_frame, resize_frame, 0, 0)
        # cv_show_wrapper(out_frame)

    else: # Default - scale to frame
        in_width, in_height, in_rgb = in_frame.shape
        scale_x, scale_y = (out_width / in_width),  (out_height / in_height)
        out_frame = cv2.resize(in_frame, None, fx = scale_x, fy = scale_y, interpolation = cv2.INTER_CUBIC)


def get_rand_eye_choice(eye_choice, ret, total_outcomes):
    # Random eye frame write - read/write from eye_choice
    if random.randint(1, total_outcomes) == 1:  # flip to other eye if success and other eye has frames
        eye_choice = "right" if eye_choice == "left" else "left"
        # logger.info(f"Eye selected: {eye_choice}")
    
    # Valid frame check - change choice if frame invalid
    if ret["right"] == False:
        eye_choice = "left"
    elif ret["left"] == False:
        eye_choice = "right"
    elif ret["right"] == False and ret["left"] == False: # error: both eyes have empty frames
        raise ValueError("Error: both eyes have ret values of False")


def edit_eyes(in_right_file, in_left_file):
    # Open the input movie file
    in_movie_right = cv2.VideoCapture(in_right_file)
    in_movie_left = cv2.VideoCapture(in_left_file)
    length_right = int(in_movie_right.get(cv2.CAP_PROP_FRAME_COUNT))
    length_left = int(in_movie_left.get(cv2.CAP_PROP_FRAME_COUNT))

    # Configure output movie file (make sure resolution/frame rate matches input video!)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore
    frame_width = out_width
    frame_height = out_height
    framerate = int(in_movie_right.get(cv2.CAP_PROP_FPS)) # TODO: how to output/respeed in/out framerate

    # Create Output movies
    out_file = os.path.join(os.getcwd(), "face_only", "right-left-edit", "edit.mp4")
    out_movie = cv2.VideoWriter(out_file, fourcc, framerate, (frame_width, frame_height))

    logger.info(f"Outfile: {out_file}")
    
    # Processing variables init
    frame_number = 0
    eye_choice = "right"
    
    # Random video overlay offset ranges init
    offset_percentage = 0.75
    offset_x = int(-(out_width * offset_percentage))
    offset_y = int(out_height + (out_height * offset_percentage))
    x_offset_right, y_offset_right = random.randint(offset_x, offset_y), random.randint(offset_x, offset_y)
    x_offset_left, y_offset_left = random.randint(offset_x, offset_y), random.randint(offset_x, offset_y)    
    
    # Processing Loop
    while frame_number<100 and (in_movie_right.isOpened() or in_movie_left.isOpened()):
        # Init blank frame for writing
        out_frame = np.zeros((frame_width, frame_height, 3), dtype=np.uint8)

        # Read a single frame of video
        frame_number += 1
        ret, in_frame = {}, {}
        ret["right"], in_frame["right"] = in_movie_right.read()
        ret["left"], in_frame["left"] = in_movie_left.read()
        
        if frame_number % 25 == 0:
            logger.info(f"Processing frame {frame_number} / R:{length_right} | L:{length_left} ")
        
        # Quit when the input video file ends
        if ret["right"] == False and ret["left"] == False:
            break

        get_rand_eye_choice(eye_choice, ret, total_outcomes=4)
        
        # Random eye offset changes
        if random.randint(1, 4) == 1:
            x_offset_right, y_offset_right = random.randint(offset_x, offset_y), random.randint(offset_x, offset_y)
            # logger.info(f"R Moving {x_offset_right}, {y_offset_right}")
        if random.randint(1, 4) == 1:
            x_offset_left, y_offset_left = random.randint(offset_x, offset_y), random.randint(offset_x, offset_y)
            # logger.info(f"L Moving {x_offset_left}, {y_offset_left}")
        
        
        # Change background or fullscreen an eye
        if random.randint(1,4) == 1:
            change_background(out_frame, in_frame[eye_choice])
        else:
            in_width, in_height, in_rgb = in_frame[eye_choice].shape
            scale_x, scale_y = (out_width / in_width),  (out_height / in_height)
            out_frame = cv2.resize(in_frame[eye_choice], None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_CUBIC)


        # Overlay Right/Left Eyes
        if ret["right"]:
            overlay_image(out_frame, in_frame["right"], x_offset_right, y_offset_right)
        if ret["left"]:
            overlay_image(out_frame, in_frame["left"], x_offset_left, y_offset_left)
        
        # Show each frame
        # if cv_show_wrapper(out_frame): break
        
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
