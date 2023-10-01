import cv2

def verify_camera_works(camera_obj):
    #input: camera object.
    #returns: Bool; whether camera returns frame.
    ret, frame = camera_obj.read()
    #print(ret)
    return ret

def grab_frame(camera_obj):
    pass

def perform_calibration_via_matrix():
    pass

if __name__ == "__main__":
    camera_index = 0

    #don't edit anything after this line.
    camera_obj = cv2.VideoCapture(camera_index)
    if not verify_camera_works(camera_obj):
        for i in range(10):#tries first 10 indexes for working camera.
            camera_obj = cv2.VideoCapture(camera_index)
            if verify_camera_works(camera_obj):
                break
    #perform_calibration_via_matrix()
            
    for i in range(100000): #not using a while true loop during development.
        frame = grab_frame(camera_obj)
        eye_box, eye_center, eye_clip = find_eye(frame)
        pupil_center = find_pupil(eye_clip)
        generate_direction(eye_center,pupil_center)
