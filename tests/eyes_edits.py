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
OUT_WIDTH = 200
OUT_HEIGHT = 200


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
        resize_frame = np.resize(in_frame, (OUT_HEIGHT, OUT_WIDTH, 3))
        resize_frame = cv2.resize(resize_frame, (OUT_HEIGHT, OUT_WIDTH), interpolation = cv2.INTER_LINEAR)
        overlay_image(out_frame, resize_frame, 0, 0)
        
    elif bg_select == 2: # np resize 75% - stutter tile
        resize_percentage = 0.75
        resize_frame = np.resize(in_frame, (int(OUT_HEIGHT * resize_percentage), int(OUT_WIDTH * resize_percentage), 3))
        resize_frame = cv2.resize(resize_frame, (OUT_HEIGHT, OUT_WIDTH), interpolation = cv2.INTER_LINEAR)
        overlay_image(out_frame, resize_frame, 0, 0)
        # cv_show_wrapper(out_frame)

    else: # Default - scale to frame
        in_width, in_height, in_rgb = in_frame.shape
        scale_x, scale_y = (OUT_WIDTH / in_width),  (OUT_HEIGHT / in_height)
        out_frame = cv2.resize(in_frame, None, fx = scale_x, fy = scale_y, interpolation = cv2.INTER_CUBIC)


def rand_flip_eye_choice(eye_choice, ret, total_outcomes):
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
    frame_width = OUT_WIDTH
    frame_height = OUT_HEIGHT
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
    offset_x = int(-(OUT_WIDTH * offset_percentage))
    offset_y = int(OUT_HEIGHT + (OUT_HEIGHT * offset_percentage))
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
        
        
        # Change background or fullscreen a random eye
        rand_flip_eye_choice(eye_choice, ret, total_outcomes=4)
        if random.randint(1,4) == 1:
            change_background(out_frame, in_frame[eye_choice])
        else:
            in_width, in_height, in_rgb = in_frame[eye_choice].shape
            scale_x, scale_y = (OUT_WIDTH / in_width),  (OUT_HEIGHT / in_height)
            out_frame = cv2.resize(in_frame[eye_choice], None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_CUBIC)


    # Overlay Right/Left Eyes
        # Random eye offset changes - 25% chance frame moves
        if random.randint(1, 4) == 1:
            x_offset_right, y_offset_right = random.randint(offset_x, offset_y), random.randint(offset_x, offset_y)
            # logger.info(f"R Moving {x_offset_right}, {y_offset_right}")
        if random.randint(1, 4) == 1:
            x_offset_left, y_offset_left = random.randint(offset_x, offset_y), random.randint(offset_x, offset_y)
            # logger.info(f"L Moving {x_offset_left}, {y_offset_left}")
            
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


def edit_grid_eyes(in_right_file, in_left_file):
    # Open the input movie file
    in_movie_right = cv2.VideoCapture(in_right_file)
    in_movie_left = cv2.VideoCapture(in_left_file)
    length_right = int(in_movie_right.get(cv2.CAP_PROP_FRAME_COUNT))
    length_left = int(in_movie_left.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Grid edit input lengths R:{length_right} | L:{length_left} ")
    
    # Configure output movie file (make sure resolution/frame rate matches input video!)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore
    frame_width = 1080
    frame_height = 1350
    framerate = int(in_movie_right.get(cv2.CAP_PROP_FPS)) # TODO: how to output/respeed in/out framerate

    out_grid_file = os.path.join(os.getcwd(), "face_only", "right-left-edit", "grid-edit.mp4")
    out_grid_movie = cv2.VideoWriter(out_grid_file, fourcc, framerate, (frame_width, frame_height))
    
    # Read in all frames of R/L video (only up to the shortest input)
    frame_number = 0
    in_frame = {"right":[], "left": []}
    
    while in_movie_right.isOpened or in_movie_left.isOpened:
        # if frame_number % 25 == 0:
        #     logger.info(f"Processing Grid frame {frame_number} / R:{length_right} | L:{length_left} ")

        # Read a single frame of video
        ret = {}
        ret["right"], right_frame = in_movie_right.read()
        ret["left"], left_frame = in_movie_left.read()
        
        # Quit when shortest video ends
        if ret["right"] == False or ret["left"] == False:
            break
        
        # Resize Frame
        right_frame = cv2.resize(right_frame, (270, 270), interpolation = cv2.INTER_LINEAR)
        left_frame = cv2.resize(left_frame, (270, 270), interpolation = cv2.INTER_LINEAR)
        
        # Add to frame list
        in_frame["right"].append(right_frame)
        in_frame["left"].append(left_frame)
        
        frame_number += 1
    
    # for i in range(5, len(in_frame["right"]), 25):
    #     if cv_show_wrapper(in_frame["right"][i]): return
    
    # Grid Edit
    frame_number = 0
    rows, cols = 5, 4
    offset_grid = generate_grid(rows, cols, 100)
    
    in_len = {}
    in_len["right"] = len(in_frame["right"])
    in_len["left"] = len(in_frame["left"])
    
    width_step = int(frame_width / cols)
    height_step = int(frame_height/ rows)
    logger.info(f"Grid: step size: {width_step} x {height_step}")
    
    eye_choice = "left"
    for frame_number in range(in_len["right"]):
        out_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        logger.info(f"Frame: {frame_number}")
        
        # Write all subframes to out_frame
        for i in range(rows):
            for j in range(cols):
                # TODO: add random R/L flip per cell?
                # rand_flip_eye_choice(eye_choice, ret, total_outcomes=4)
                
                subframe_x, subframe_y = i * width_step, j * height_step
                subframe_offset = (frame_number + offset_grid[i][j]) % in_len[eye_choice]
                
                subframe = in_frame[eye_choice][subframe_offset]
                # resize_frame = cv2.resize(subframe, (width_step, height_step), interpolation = cv2.INTER_LINEAR)
                # logger.info(f"Subframe ({i},{j}): Dimension:({subframe_x},{subframe_y}) ")
                overlay_image(out_frame, subframe, subframe_x, subframe_y)

        # if cv_show_wrapper(out_frame): return
        out_grid_movie.write(out_frame)
    
 

if __name__ == "__main__":
    logger.info("=== Beginning Edits ===")
    
    # File names
    right_vid = "right.avi"
    left_vid = "left.avi"
    
    # Get file from the test dir
    face_dir = os.path.join(os.getcwd(), "face_only", "right-left-edit")
    right_file = os.path.join(face_dir, right_vid)
    left_file = os.path.join(face_dir, left_vid)
    # logger.info(f"Editing Files: \nR: {right_file} \nL: {left_file}")
    # edit_eyes(right_file, left_file)
    
    right_vid = "still-girl-face-right.mp4"
    left_vid = "still-girl-face-left.mp4"
    logger.info(f"Editing Files: \nR: {right_file} \nL: {left_file}")
    
    face_dir = os.path.join(os.getcwd(), "face_only", "right-left-edit")
    right_file = os.path.join(face_dir, right_vid)
    left_file = os.path.join(face_dir, left_vid)
    edit_grid_eyes(right_file, left_file)

    
    cv2.destroyAllWindows()
