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

def find_pupils_blob(eye_clips, blob_detector):
    #input: face cropped
    #returns: pupil locations.
    crop_ratio = 0.12 #crops 10% from left, right, top, and bottom.
    thresholds = [70,140]
    threshold_increment = 20
    blob_locations = []
    temp_thresh = 140
    left_pupil_deviation = []
    right_pupil_deviation = []
    for temp_thresh in range(thresholds[0], thresholds[1], 10):
        for eye_clip in eye_clips:
            #print(eye_clip[0].shape)
            temp_eye_clip = eye_clip[0]

            #crops the eye further
            temp_eye_clip = temp_eye_clip[int(len(temp_eye_clip)*crop_ratio):int(len(temp_eye_clip)*(1-crop_ratio)),
                                int(len(temp_eye_clip[0])*crop_ratio):int(len(temp_eye_clip[0])*(1-crop_ratio)),:]
            #print(temp_eye_clip.shape)
            eye_clip_img = cv2.cvtColor(temp_eye_clip, cv2.COLOR_BGR2GRAY)
            ret, eye_clip_thresh = cv2.threshold(eye_clip_img, temp_thresh, 255, cv2.THRESH_BINARY)
            data = blob_detector.detect(eye_clip_thresh)
            #print(data)
            #print(eye_clip_thresh.shape)
            cv2.drawKeypoints(temp_eye_clip, data, temp_eye_clip, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imshow("thresh",temp_eye_clip)
            cv2.waitKey(5)

            #if left pupil:
            y, y_h, total_h, x, x_w, total_w = eye_clip[1]
            
            try:
                blob_x = data[0].pt[0]
                blob_y = data[0].pt[1]
                if (x+x_w)/2 < total_w/2: #on left
                    #print(f"left")
                    #left_pupil_data.append([blob_x, blob_y])
                    left_pupil_x_deviation = 1 - (blob_x)/ len(eye_clip_thresh)
                    left_pupil_y_deviation = 1 - (blob_y) / len(eye_clip_thresh[0])
                    left_pupil_deviation.append([left_pupil_x_deviation, left_pupil_y_deviation])
                    #print("GOT TO HERE1")
                else:
                    #print(f"right")
                    #right_pupil_data.append([blob_x, blob_y])
                    right_pupil_x_deviation = 1 - (blob_x)/ len(eye_clip_thresh)
                    right_pupil_y_deviation = 1 - (blob_y)/ len(eye_clip_thresh[0])
                    right_pupil_deviation.append([right_pupil_x_deviation, right_pupil_y_deviation])
                    #print("GOT TO HERE2")
                #print(blob_x)
                
            except:
                pass
            #left pupil x,y
    try:
        avg_left_pupil_x_deviation = sum([x for x,y in left_pupil_deviation])/len(left_pupil_deviation)
        avg_left_pupil_y_deviation = sum([y for x,y in left_pupil_deviation])/len(left_pupil_deviation)
    except:
        avg_left_pupil_x_deviation = None
        avg_left_pupil_y_deviation = None

    try:
        avg_right_pupil_x_deviation = sum([x for x,y in right_pupil_deviation])/len(right_pupil_deviation)
        avg_right_pupil_y_deviation = sum([y for x,y in right_pupil_deviation])/len(right_pupil_deviation)
    except:
        avg_right_pupil_x_deviation = None
        avg_right_pupil_y_deviation = None

    if verbose:
        print(avg_left_pupil_x_deviation, avg_left_pupil_y_deviation, avg_right_pupil_x_deviation, avg_left_pupil_y_deviation)
    
    return avg_left_pupil_x_deviation, avg_left_pupil_y_deviation, avg_right_pupil_x_deviation, avg_right_pupil_y_deviation

def generate_direction(left_pupil_x, left_pupil_y, right_pupil_x, right_pupil_y, left_eye_matrix, right_eye_matrix, verbose):
    estimated_left_dot_x = None
    estimated_left_dot_y = None
    estimated_right_dot_x = None
    estimated_right_dot_y = None
    if left_pupil_x is None or left_pupil_y is None or right_pupil_x is None or right_pupil_y is None:
        if verbose:
            print("Pupil positions are None!")
        return None, None
    
    def apply_transformation(x, y, matrix):
        point = np.array([[x], [y], [1]])
        transformed_point = np.dot(matrix, point)
        return transformed_point[0, 0], transformed_point[1, 0]
    if not (left_pupil_x is None or left_pupil_y is None):
        estimated_left_dot_x, estimated_left_dot_y = apply_transformation(left_pupil_x, left_pupil_y, left_eye_matrix[0])
    if not (right_pupil_x is None or right_pupil_y is None):
        estimated_right_dot_x, estimated_right_dot_y = apply_transformation(right_pupil_x, right_pupil_y, right_eye_matrix[0])
    
    if verbose:
        print("Estimated Left Dot Position: ", estimated_left_dot_x, estimated_left_dot_y)
        print("Estimated Right Dot Position: ", estimated_right_dot_x, estimated_right_dot_y)
    
    return (estimated_left_dot_x, estimated_left_dot_y), (estimated_right_dot_x, estimated_right_dot_y)



def find_transformation_matrix(eye_offsets, dot_positions, verbose):
    # Remove data points where any value is None
    clean_data = [(eo, dp) for eo, dp in zip(eye_offsets, dot_positions) if None not in eo and None not in dp]
    
    # Check if we have enough valid data points
    if len(clean_data) < 3:
        if verbose:
            print("Not enough valid data points for calibration!")
        return None
    
    eye_offsets_clean, dot_positions_clean = zip(*clean_data)
    eye_offsets_clean = np.array(eye_offsets_clean)
    dot_positions_clean = np.array(dot_positions_clean)
    
    transformation_matrices = []
    
    for i in range(2):
        model = LinearRegression().fit(eye_offsets_clean, dot_positions_clean[:, i])
        transformation_matrix = np.zeros((3, 3))
        transformation_matrix[0, :-1] = model.coef_
        transformation_matrix[0, -1] = model.intercept_
        transformation_matrix[1, 1] = 1
        transformation_matrix[2, 2] = 1
        transformation_matrices.append(transformation_matrix)
        
    return transformation_matrices

def perform_calibration_via_matrix(calibration_steps, verbose):
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

def plot_results(estimated_left_dot_position, estimated_right_dot_position):
    res_x = 1920
    res_y = 1080
    
    #print(rand_x, rand_y)
    display_text = f"Please look at the red and press any key.\n If no key is pressed, the next image will be shown in 10 seconds"
    cv2_image = 60*np.ones((res_x, res_y, 3), dtype=np.uint8)
    try:
        value_right_x = int(res_x * estimated_right_dot_position[0])
        value_right_y = int(res_y * estimated_right_dot_position[1])
        cv2.circle(cv2_image, (value_right_y, value_right_x), 50, (0, 0, 255), -1)
    except:
        pass
    try:
        value_left_x = int(res_x * estimated_left_dot_position[0])
        value_left_y = int(res_y * estimated_left_dot_position[1])
        cv2.circle(cv2_image, (value_left_y, value_left_x), 50, (0, 0, 255), -1)
    except:
        pass
    
    
    #cv2_image[]
    cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    cv2.imshow("window", cv2_image) 


if __name__ == "__main__":
    camera_index = 0
    face_haarcascade_filename = "haarcascade_frontalface_default.xml"
    eye_haarcascade_filename = "haarcascade_eye.xml"
    verbose = True
    calibration_steps = 10
    
    #don't edit anything after this line.
    camera_obj = cv2.VideoCapture(camera_index)
    face_hc = cv2.CascadeClassifier(cv2.data.haarcascades + face_haarcascade_filename)
    eye_hc = cv2.CascadeClassifier(cv2.data.haarcascades + eye_haarcascade_filename)
    blob_detector_parameters = cv2.SimpleBlobDetector_Params()
    blob_detector_parameters.filterByCircularity = True
    blob_detector_parameters.minCircularity = 0.05
    blob_detector_parameters.filterByArea = True
    blob_detector_parameters.maxArea = 10000
    blob_detector_parameters.filterByConvexity = True
    blob_detector_parameters.minConvexity = 0.5
    blob_detector_parameters.filterByInertia = True
    blob_detector_parameters.minInertiaRatio = 0.01
    blob_detector = cv2.SimpleBlobDetector_create(blob_detector_parameters)
    
    if not verify_camera_works(camera_obj):
        for i in range(10):#tries first 10 indexes for working camera.
            camera_obj = cv2.VideoCapture(camera_index)
            if verify_camera_works(camera_obj):
                break
    left_eye_matrix, right_eye_matrix = perform_calibration_via_matrix(calibration_steps, verbose)
            
    for i in range(100000): #not using a while true loop during development.
        frame = grab_frame(camera_obj)
        eye_clips = find_eye(frame, face_hc, eye_hc, verbose)
        left_pupil_x, left_pupil_y, right_pupil_x, right_pupil_y = find_pupils_blob(eye_clips, blob_detector)
        estimated_left_dot_position, estimated_right_dot_position = generate_direction(left_pupil_x, left_pupil_y, right_pupil_x, right_pupil_y, left_eye_matrix, right_eye_matrix, verbose)
        print(estimated_left_dot_position, estimated_right_dot_position)
        plot_results(estimated_left_dot_position, estimated_right_dot_position)
