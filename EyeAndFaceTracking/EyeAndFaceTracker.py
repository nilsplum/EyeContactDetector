import logging
import numpy as np
import threading
import math
import cv2
import mediapipe as mp
import time


class FaceAndEyeTracker:
    """
        This class does the real-time image processing of the camera feed using opencv to extract the following data:
        - Pixeldata around both eyes
        - 3D head position

        Attributes:
        - lock (threading.Lock): A threading lock to ensure thread-safe access and modification of shared resources, specifically the current frame `self.img`.
        - mpFaceMesh (mp.solutions.face_mesh): The MediaPipe Face Mesh solution for face and eye landmark detection.
        - faceMesh: An instance of the Face Mesh detector configured with specific options for maximum number of faces and detection confidences.
        - mpDraw (mp.solutions.drawing_utils): Utility for drawing MediaPipe landmarks on images.
        - drawSpec: Specifications for drawing landmarks, including thickness and circle radius.
        - l_facial_data (list): Stores calculated head pose data and the distance between lips.
        - img: The current frame captured from the video feed.

        The primary method, `eye_and_facetracking_thread`, orchestrates the capture, processing, and analysis of video frames
        to detect facial landmarks, estimate head pose, and derive eye contact metrics.
    """

    def __init__(self):
        """
            Initializes the FaceAndEyeTracker object, setting up the MediaPipe solutions, drawing specifications,
            and preparing threading locks for safe concurrent access to the frame data.
        """
        self.lock = threading.Lock()
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(max_num_faces=2, min_tracking_confidence=0.5,
                                                 min_detection_confidence=0.5)
        self.mpDraw = mp.solutions.drawing_utils
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)
        self.l_facial_data = [None, None, None]  # Stores the head pose and distance between lips information
        self.img = None  # This will hold the current frame

    def init_eye_tracker(self, thread=True):
        if thread:
            x = threading.Thread(target=self.eye_and_facetracking_thread)
            x.start()
        else:
            self.eye_and_facetracking_thread(True)

    def eye_and_facetracking_thread(self, output_opencv=False):
        cap = cv2.VideoCapture(0)
        p_time = 0

        while True:
            try:
                success, temp_img = cap.read()
                if success:
                    try:
                        imgRGB = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB)
                    except Exception as e:
                        logging.error(
                            "Something went wrong with connecting to the camera. \n for macos please check if 'Continuity Camera' is enabled")
                        continue
                    temp_img.flags.writeable = False
                    results = self.faceMesh.process(imgRGB)
                    temp_img.flags.writeable = True
                    img_h, img_w, img_c = temp_img.shape
                    face_3d = []
                    face_2d = []

                    if results.multi_face_landmarks:
                        for face_landmarks in results.multi_face_landmarks:
                            f_distance_lips = None
                            lm_lips_top = None
                            lm_lips_bottom = None
                            for id, lm in enumerate(face_landmarks.landmark):
                                if id in [33, 263, 1, 61, 291, 199]:
                                    if id == 1:
                                        nose_2d = (lm.x * img_w, lm.y * img_h)

                                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                                    face_2d.append([x, y])
                                    face_3d.append([x, y, lm.z])

                                if id == 13:
                                    lm_lips_top = int(lm.x * img_w), int(lm.y * img_h)
                                if id == 14:
                                    lm_lips_bottom = int(lm.x * img_w), int(lm.y * img_h)

                            focal_length = 1 * img_w
                            cam_matrix = np.array(
                                [[focal_length, 0, img_h / 2], [0, focal_length, img_w / 2], [0, 0, 1]])
                            dist_matrix = np.zeros((4, 1), dtype=np.float64)
                            success, rot_vec, trans_vec = cv2.solvePnP(np.array(face_3d, dtype=np.float64),
                                                                       np.array(face_2d, dtype=np.float64), cam_matrix,
                                                                       dist_matrix)
                            rmat, jac = cv2.Rodrigues(rot_vec)
                            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
                            x = angles[0] * 360
                            y = angles[1] * 360
                            point1 = (int(nose_2d[0]), int(nose_2d[1]))
                            point2 = (int(nose_2d[0] + y * 60), int(nose_2d[1] - x * 60))
                            self.l_facial_data[0] = math.sqrt((point1[0] - point2[0]) ** 2)
                            self.l_facial_data[1] = math.sqrt((point1[1] - point2[1]) ** 2)
                            self.l_facial_data[2] = lm_lips_bottom[1] - lm_lips_top[1]

                            self.landmarks_left_eye_nparray = self.get_nparray_from_landmarks("left", face_landmarks,
                                                                                              temp_img)
                            self.landmarks_right_eye_nparray = self.get_nparray_from_landmarks("right", face_landmarks,
                                                                                               temp_img)

                            self.mpDraw.draw_landmarks(
                                image=temp_img,
                                landmark_list=face_landmarks,
                                connections=self.mpFaceMesh.FACEMESH_CONTOURS,
                                landmark_drawing_spec=self.drawSpec,
                                connection_drawing_spec=self.drawSpec)
                            if self.landmarks_left_eye_nparray is not None and self.landmarks_right_eye_nparray is not None:
                                cv2.polylines(temp_img, [self.landmarks_left_eye_nparray], True, (0, 255, 0), 2)
                                cv2.polylines(temp_img, [self.landmarks_right_eye_nparray], True, (0, 255, 0), 2)

                            cv2.line(temp_img, point1, point2, (0, 255, 0), 3)
                            cv2.line(temp_img, lm_lips_bottom, lm_lips_top, (0, 255, 0), 3)
                    else:
                        self.landmarks_left_eye_nparray = None
                        self.landmarks_right_eye_nparray = None

                    c_time = time.time()
                    fps = 1 / (c_time - p_time)
                    p_time = c_time
                    cv2.putText(temp_img, f'FPS:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    if output_opencv:
                        cv2.imshow("Test", temp_img)
                        cv2.waitKey(1)

                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                    with self.lock:
                        self.img = temp_img
            except Exception as e:
                logging.error(f"Error in EyeAndFaceTracker.get_current_eye_pixels: \n{e}")

        cap.release()

    def get_current_face_data(self):
        """
            Safely retrieves the latest facial data processed by the tracker. This includes head pose information and
            the distance between the lips (for potential detection if the person speaks to the camera),
            encapsulating key metrics for head tilt information
            Returns:
            - list: The latest head pose data and distance between lips, encapsulated in a list.
        """
        try:
            return self.l_facial_data
        except Exception as e:
            logging.error(f"Error in EyeAndFaceTracker.get_current_face_data: \n{e}")

    def get_current_eye_pixels(self) -> (np.ndarray, np.ndarray):
        """
            Extracts and returns the pixel data for the left and right eyes based on the detected landmarks. This method
            ensures thread-safe access to the current frame and processes it to extract eye regions for further analysis.

            Returns:
            - tuple: A tuple containing NumPy arrays for the left and right eye regions. Returns `None, None` if eye
                     landmarks are not available or an error occurs during processing.
        """
        with self.lock:
            temp_landmarks_left_eye_nparray = self.landmarks_left_eye_nparray
            temp_landmarks_right_eye_nparray = self.landmarks_right_eye_nparray
        if temp_landmarks_left_eye_nparray is not None and temp_landmarks_right_eye_nparray is not None:
            try:
                height, width, _ = self.img.shape
                mask = np.zeros((height, width), np.uint8)
                cv2.polylines(mask, [temp_landmarks_left_eye_nparray], True, 255, 2)
                cv2.fillPoly(mask, [temp_landmarks_left_eye_nparray], 255)
                temp_eye_left = cv2.bitwise_and(self.img, self.img, mask=mask)

                min_x_left = np.min(temp_landmarks_left_eye_nparray[:, 0])
                max_x_left = np.max(temp_landmarks_left_eye_nparray[:, 0])
                min_y_left = np.min(temp_landmarks_left_eye_nparray[:, 1])
                max_y_left = np.max(temp_landmarks_left_eye_nparray[:, 1])
                gray_eye_left = temp_eye_left[min_y_left: max_y_left, min_x_left: max_x_left]
                _, threshold_eye_left = cv2.threshold(gray_eye_left, 70, 255, cv2.THRESH_BINARY)
                eye_left = cv2.resize(gray_eye_left, None, fx=10, fy=10)

                cv2.polylines(mask, [temp_landmarks_right_eye_nparray], True, 255, 2)
                cv2.fillPoly(mask, [temp_landmarks_right_eye_nparray], 255)
                temp_eye_right = cv2.bitwise_and(self.img, self.img, mask=mask)

                min_x_right = np.min(temp_landmarks_right_eye_nparray[:, 0])
                max_x_right = np.max(temp_landmarks_right_eye_nparray[:, 0])
                min_y_right = np.min(temp_landmarks_right_eye_nparray[:, 1])
                max_y_right = np.max(temp_landmarks_right_eye_nparray[:, 1])
                gray_eye_right = temp_eye_right[min_y_right: max_y_right, min_x_right: max_x_right]
                _, threshold_eye_right = cv2.threshold(gray_eye_right, 70, 255, cv2.THRESH_BINARY)
                eye_right = cv2.resize(gray_eye_right, None, fx=10, fy=10)
                return eye_left, eye_right
            except Exception as e:
                logging.error(f"Error in EyeAndFaceTracker.get_current_eye_pixels: \n{e}")
        else:
            return None, None

    def get_nparray_from_landmarks(self, left_or_right_eye, landmarks, img):
        """
            Converts detected landmarks for a specified eye into a NumPy array format. This utility method supports the
            extraction of specific eye regions by organizing relevant landmark points into structured arrays.

            Parameters:
            - left_or_right_eye (str): A string indicating which eye's landmarks to process ('left' or 'right').
            - landmarks: The detected landmarks from MediaPipe Face Mesh for a single face.
            - img: The current frame on which landmarks were detected.

            Returns:
            - np.array: A NumPy array containing the specified eye's landmarks, formatted for further processing.
        """
        # Creates nparray with shape (16,2) and numbers that get overwritten
        landmarks_eye_nparray = np.arange(34, ).reshape(17, 2)
        ih, iw, ic = img.shape

        if left_or_right_eye == "left":
            landmarks_eye_nparray[0][0] = iw * landmarks.landmark[244].x
            landmarks_eye_nparray[0][1] = ih * landmarks.landmark[244].y

            landmarks_eye_nparray[1][0] = iw * landmarks.landmark[189].x
            landmarks_eye_nparray[1][1] = ih * landmarks.landmark[189].y

            landmarks_eye_nparray[2][0] = iw * landmarks.landmark[221].x
            landmarks_eye_nparray[2][1] = ih * landmarks.landmark[221].y

            landmarks_eye_nparray[3][0] = iw * landmarks.landmark[222].x
            landmarks_eye_nparray[3][1] = ih * landmarks.landmark[222].y

            landmarks_eye_nparray[4][0] = iw * landmarks.landmark[223].x
            landmarks_eye_nparray[4][1] = ih * landmarks.landmark[223].y

            landmarks_eye_nparray[5][0] = iw * landmarks.landmark[224].x
            landmarks_eye_nparray[5][1] = ih * landmarks.landmark[224].y

            landmarks_eye_nparray[6][0] = iw * landmarks.landmark[225].x
            landmarks_eye_nparray[6][1] = ih * landmarks.landmark[225].y

            landmarks_eye_nparray[7][0] = iw * landmarks.landmark[113].x
            landmarks_eye_nparray[7][1] = ih * landmarks.landmark[113].y

            landmarks_eye_nparray[8][0] = iw * landmarks.landmark[130].x
            landmarks_eye_nparray[8][1] = ih * landmarks.landmark[130].y

            landmarks_eye_nparray[9][0] = iw * landmarks.landmark[25].x
            landmarks_eye_nparray[9][1] = ih * landmarks.landmark[25].y

            landmarks_eye_nparray[10][0] = iw * landmarks.landmark[110].x
            landmarks_eye_nparray[10][1] = ih * landmarks.landmark[110].y

            landmarks_eye_nparray[11][0] = iw * landmarks.landmark[24].x
            landmarks_eye_nparray[11][1] = ih * landmarks.landmark[24].y

            landmarks_eye_nparray[12][0] = iw * landmarks.landmark[23].x
            landmarks_eye_nparray[12][1] = ih * landmarks.landmark[23].y

            landmarks_eye_nparray[13][0] = iw * landmarks.landmark[22].x
            landmarks_eye_nparray[13][1] = ih * landmarks.landmark[22].y

            landmarks_eye_nparray[14][0] = iw * landmarks.landmark[26].x
            landmarks_eye_nparray[14][1] = ih * landmarks.landmark[26].y

            landmarks_eye_nparray[15][0] = iw * landmarks.landmark[112].x
            landmarks_eye_nparray[15][1] = ih * landmarks.landmark[112].y

            landmarks_eye_nparray[16][0] = iw * landmarks.landmark[243].x
            landmarks_eye_nparray[16][1] = ih * landmarks.landmark[243].y

        if left_or_right_eye == "right":
            landmarks_eye_nparray[0][0] = iw * landmarks.landmark[464].x
            landmarks_eye_nparray[0][1] = ih * landmarks.landmark[464].y

            landmarks_eye_nparray[1][0] = iw * landmarks.landmark[413].x
            landmarks_eye_nparray[1][1] = ih * landmarks.landmark[413].y

            landmarks_eye_nparray[2][0] = iw * landmarks.landmark[441].x
            landmarks_eye_nparray[2][1] = ih * landmarks.landmark[441].y

            landmarks_eye_nparray[3][0] = iw * landmarks.landmark[442].x
            landmarks_eye_nparray[3][1] = ih * landmarks.landmark[442].y

            landmarks_eye_nparray[4][0] = iw * landmarks.landmark[443].x
            landmarks_eye_nparray[4][1] = ih * landmarks.landmark[443].y

            landmarks_eye_nparray[5][0] = iw * landmarks.landmark[444].x
            landmarks_eye_nparray[5][1] = ih * landmarks.landmark[444].y

            landmarks_eye_nparray[6][0] = iw * landmarks.landmark[445].x
            landmarks_eye_nparray[6][1] = ih * landmarks.landmark[445].y

            landmarks_eye_nparray[7][0] = iw * landmarks.landmark[342].x
            landmarks_eye_nparray[7][1] = ih * landmarks.landmark[342].y

            landmarks_eye_nparray[8][0] = iw * landmarks.landmark[359].x
            landmarks_eye_nparray[8][1] = ih * landmarks.landmark[359].y

            landmarks_eye_nparray[9][0] = iw * landmarks.landmark[255].x
            landmarks_eye_nparray[9][1] = ih * landmarks.landmark[255].y

            landmarks_eye_nparray[10][0] = iw * landmarks.landmark[339].x
            landmarks_eye_nparray[10][1] = ih * landmarks.landmark[339].y

            landmarks_eye_nparray[11][0] = iw * landmarks.landmark[254].x
            landmarks_eye_nparray[11][1] = ih * landmarks.landmark[254].y

            landmarks_eye_nparray[12][0] = iw * landmarks.landmark[253].x
            landmarks_eye_nparray[12][1] = ih * landmarks.landmark[253].y

            landmarks_eye_nparray[13][0] = iw * landmarks.landmark[252].x
            landmarks_eye_nparray[13][1] = ih * landmarks.landmark[252].y

            landmarks_eye_nparray[14][0] = iw * landmarks.landmark[256].x
            landmarks_eye_nparray[14][1] = ih * landmarks.landmark[256].y

            landmarks_eye_nparray[15][0] = iw * landmarks.landmark[341].x
            landmarks_eye_nparray[15][1] = ih * landmarks.landmark[341].y

            landmarks_eye_nparray[16][0] = iw * landmarks.landmark[463].x
            landmarks_eye_nparray[16][1] = ih * landmarks.landmark[463].y

        return landmarks_eye_nparray
