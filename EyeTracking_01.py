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
        cropped_eyes = [cropped_face[y:y+h, x:x+w] for (x, y, w, h) in eyes]
        all_eyes.extend(cropped_eyes)
    if verbose:
        try:
            cv2.imshow("eye1",all_eyes[0])
            cv2.waitKey(5)
        except:
            pass
    if verbose:
        print(f"{len(faces)} faces detected. {len(all_eyes)} eyes detected.")
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
    verbose = True
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
        eye_box, eye_center, eye_clip = find_eye(frame, face_hc, eye_hc, verbose)
        pupil_center = find_pupil(eye_clip)
        generate_direction(eye_center,pupil_center)
