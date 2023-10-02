import cv2
import numpy as np
import random
from sklearn.linear_model import LinearRegression

def verify_camera_works(camera_obj):
    #input: camera object.
    #returns: Bool; whether camera returns frame.
    ret, frame = camera_obj.read()
    #print(ret)
    return ret

def grab_frame(camera_obj):
    ret, frame = camera_obj.read()
    return frame

def find_eye(frame, face_hc, eye_hc, verbose):
    #input: frame.
    #returns: cropped image of eye.

    #crop image to head to minimize risk of false eyes.
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("test", gray_frame)
    cv2.waitKey(5)
    #print(gray_frame)
    faces = face_hc.detectMultiScale(gray_frame, 1.1, 5, minSize=(30, 30))
    #print(faces)
    
    cropped_faces = [frame[y:y+h, x:x+w] for (x, y, w, h) in faces]
    if verbose:
        try:
            cv2.imshow("face1",cropped_faces[0])
            cv2.waitKey(5)
        except:
            pass

    all_eyes = []
    for cropped_face in cropped_faces:
        eyes = eye_hc.detectMultiScale(cropped_face, 1.1, 5, minSize=(7, 7))
        cropped_eyes = [[cropped_face[y:y+h, x:x+w], [y,y+h, len(cropped_face), x,x+w, len(cropped_face[0])]] for (x, y, w, h) in eyes]
        all_eyes.extend(cropped_eyes)
    if verbose:
        try:
            cv2.imshow("eye1",all_eyes[0][0])
            cv2.waitKey(5)
        except:
            pass
    #if verbose:
        #print(f"{len(faces)} faces detected. {len(all_eyes)} eyes detected.")

    
    return all_eyes



def perform_calibration(calibration_steps, verbose):
    dot_positions = []
    left_eye_offsets = []
    right_eye_offsets = []
    for i in range(calibration_steps):
        res_x = 1920
        res_y = 1080
        rand_x = random.uniform(0.02,0.98)
        rand_y = random.uniform(0.02,0.98)
        #print(rand_x, rand_y)
        display_text = f"Please look at the red and press any key.\n If no key is pressed, the next image will be shown in 10 seconds"
        cv2_image = 90*np.ones((res_x, res_y, 3), dtype=np.uint8)
        cv2.circle(cv2_image, (int(res_y*rand_y), int(res_x*rand_x)), 5, (0, 0, 255), -1)
        #cv2_image[]
        cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
        cv2.imshow("window", cv2_image)

        #when a key is pressed, terminate the frame and look for eyes.
        cv2.waitKey(9999)
        frame = grab_frame(camera_obj)
        eye_clips = find_eye(frame, face_hc, eye_hc, verbose)
        left_pupil_x, left_pupil_y, right_pupil_x, right_pupil_y = find_pupils_blob(eye_clips, blob_detector)
        print(left_pupil_x, left_pupil_y, right_pupil_x, right_pupil_y)
        #add the collected data to an array.

        dot_positions.append([rand_x, rand_y])
        left_eye_offsets.append([left_pupil_x, left_pupil_y])
        right_eye_offsets.append([right_pupil_x, right_pupil_y])
    cv2.destroyAllWindows()

    #generate transformation matrix to convert eye offsets to an estimated dot position.
    left_eye_matrix = find_transformation_matrix(left_eye_offsets, dot_positions, verbose)
    right_eye_matrix = find_transformation_matrix(right_eye_offsets, dot_positions, verbose)
    if left_eye_matrix is None or right_eye_matrix is None:
        if verbose:
            print("Calibration failed due to insufficient valid data points!")
        return None, None
    
    return left_eye_matrix, right_eye_matrix



if __name__ == "__main__":
    camera_index = 0
    face_haarcascade_filename = "haarcascade_frontalface_default.xml"
    eye_haarcascade_filename = "haarcascade_eye.xml"
    verbose = True
    calibration_steps = 200
    
    if not verify_camera_works(camera_obj):
        for i in range(10):#tries first 10 indexes for working camera.
            camera_obj = cv2.VideoCapture(camera_index)
            if verify_camera_works(camera_obj):
                break
    perform_calibration(calibration_steps, verbose)
