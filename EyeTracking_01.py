import cv2
import numpy

def verify_camera_works(camera_obj):
    #input: camera object.
    #returns: Bool; whether camera returns frame.
    ret, frame = camera_obj.read()
    #print(ret)
    return ret

def grab_frame(camera_obj):
    ret, frame = camera_obj.read()
    return frame

def find_eye(frame, face_hc, eye_hc):
    #input: frame.
    #returns: cropped image of eye.

    #crop image to head to minimize risk of false eyes.
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("test", gray_frame)
    cv2.waitKey(1)
    #print(gray_frame)
    faces = face_cascade.detectMultiScale(gray_frame, 1.5, 5)
    print(faces)
    return False, False, False

def find_pupil(eye_clip):
    #input: face cropped
    #returns: pupil locations.
    pass

def generate_direction(eye_center, pupil_center):
    pass

def perform_calibration_via_matrix():
    pass

if __name__ == "__main__":
    camera_index = 0
    face_haarcascade_filename = "haarcascade_frontalface_default.xml"
    eye_haarcascade_filename = "haarcascade_eye.xml"
    #don't edit anything after this line.
    camera_obj = cv2.VideoCapture(camera_index)
    face_hc = cv2.CascadeClassifier(cv2.data.haarcascades + face_haarcascade_filename)
    eye_hc = cv2.CascadeClassifier(cv2.data.haarcascades + eye_haarcascade_filename)
    if not verify_camera_works(camera_obj):
        for i in range(10):#tries first 10 indexes for working camera.
            camera_obj = cv2.VideoCapture(camera_index)
            if verify_camera_works(camera_obj):
                break
    #perform_calibration_via_matrix()
            
    for i in range(100000): #not using a while true loop during development.
        frame = grab_frame(camera_obj)
        eye_box, eye_center, eye_clip = find_eye(frame, face_hc, eye_hc)
        pupil_center = find_pupil(eye_clip)
        generate_direction(eye_center,pupil_center)
