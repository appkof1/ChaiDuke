import os

import torch
import cv2
import numpy as np
from ultralytics import YOLO #object detection and pose estimation

from tqdm import tqdm #used for tracking progress

from strongsort import StrongSORT #Object tracker
from pathlib import Path

from ClassifyReversibility import ReversibilityClassifier #Pytorch neural network for classifying player movement
from ClassifyReversibility import basic_classify #basic reversibility classifying function

"""
Optical flow parameters mostly kept the same as from source documentation
"""
feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)

lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.03))

classify_reverse = ReversibilityClassifier()
classify_reverse.load_state_dict(torch.load("final_model.pth"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classify_reverse = classify_reverse.to(device)

tracker = StrongSORT(model_weights=Path('osnet_x0_25_imagenet.pt'), device='cuda', fp16=False,
                     max_iou_distance=0.9, max_age=6)

test_files_dir = "football_video_folder"
test_files = os.listdir(test_files_dir)

n_videos = len(test_files)

from joblib import dump, load
norm_scale_pipeline = load("preprocessing_pipeline.pkl") #normalizes then scales data based on the training data it fitted to


"""
YOLO object and pose detection setup
"""
det_model_path = "object_detector.pt"
det_model = YOLO(det_model_path, task="detect")
det_names = det_model.names
det_model.to(0)
pose_model_path = "yolov8n-pose.pt"
pose_model = YOLO(pose_model_path)

reversed_actual = False


def getTrackXYXY(track_id, tracksXYXY): #from a track id returns the xyxy format bounding box coordinates
    for track in tracksXYXY:
        if track[4] == track_id:
            return track[:4]

"""
These are the lists which the files and their reversibility scores will be sorted into
"""
true_positives = []
true_negatives = []
false_positives = []
false_negatives = []


classification_methods = ["basic","neural network"]
chosen_classification_method = classification_methods[1] # 0 for basic, 1 for neural network classification

for i in tqdm(range(n_videos), leave=False, position=0): #this just produces a progress bar to visualize how far through it is
    video_path = f"{test_files_dir}/{test_files[i]}" #selects next video
    if "reversed" in video_path:
        reversed_actual = True
    else:
        reversed_actual = False

    cap = cv2.VideoCapture(video_path)
    ret = True
    p0 = []
    predict_reversed = 0
    predict_not_reversed = 0

    while ret:
        try:
            ret, frame = cap.read()
            if not ret:
                break
        except:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #converts image to grayscale for optical flow
        mask = np.ones_like(frame_gray) #sets a default mask of all 1s

        det_output = det_model.predict(frame, conf=0.5, verbose=False)
        detections = det_output[0] #gets the detections from the frame
        confs = detections.boxes.conf.tolist() #detection confidences
        classes = detections.boxes.cls.tolist() #detection classes

        boxes_xyxy = detections.boxes.xyxy.tolist()
        for bbox in boxes_xyxy:
            mask[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] = 0 #sets the area of each bounding box in the mask to 0s

        if len(p0) == 0: #if no more points found to track for optical flow, find new ones
            old_gray = frame_gray.copy()
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)

        if p0 is not None:
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params) #find the next position of the tracked points

        if p1 is not None:
            optical_flow = p1 - p0
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            pixel_velocities = optical_flow.mean(axis=0) #gets the average velocities of pixels in the x and y dimensions
            p0 = good_new.reshape(-1, 1, 2)

        old_gray = frame_gray.copy()

        tracks_xyxy = []
        tracks_xyxy = tracker.update(detections.boxes.data.cpu(), ori_img=frame) #update the object tracker with detections

        tracks = []
        for track in tracker.tracker.tracks:
            if track.state == 2: #if the track is confirmed, i.e. detected for enough frames in a row, get the information for each track and store it in a list
                track_mean = track.mean
                track_det_class = np.array([track.class_id])
                track_id = np.array([track.track_id])
                track_conf = np.array([track.conf])
                tracks.append(np.concatenate([track_mean, track_det_class, track_conf, track_id]))


        track_of_ball= []
        highest_ball_conf = 0
        """
        Gets the track of the ball.
        If there are somehow multiple tracks where a ball is detected, choose the highest confidence one 
        """
        for track in tracks:
            track_class = track[8]
            conf = track[9]
            if track_class == 0:
                if conf > highest_ball_conf:
                    highest_ball_conf = conf
                    track_of_ball = track

        if len(track_of_ball) > 0:
            ball_data = [track_of_ball[0],track_of_ball[1],track_of_ball[4],track_of_ball[5]] #gets the ball's x,y locations and velocities
        else:
            ball_data = [0, 0, 0, 0] #if the track for the ball is not found, set the array to 0s

        ball_data = np.array(ball_data)

        pose_predictions = pose_model.predict(frame, imgsz=1920, conf=0.2, verbose=False) #find the poses of people in the frame

        pose_tracks = []


        for i in range(len(pose_predictions[0].boxes)):
            """
            For each detected pose in the frame, this section of code tries to match it to a track
            If a track is found for it, it is added to a list which stores pairs of poses and track data
            """
            box = pose_predictions[0].boxes.xyxy[i] #get the bounding boxes for the people detected by the pose estimator
            if p1 is not None:
                smallestOffset = 100
                closestTrackPair = []
                for track in tracks:
                    track_class = track[8]
                    if track_class == 2:  # track detection class
                        offset = 0
                        track_xyxy = getTrackXYXY(track[10], tracks_xyxy)
                        if track_xyxy is not None:
                            for j in range(4):
                                offset += abs(box.cpu()[j] - track_xyxy[j])

                            if offset <= smallestOffset:
                                smallestOffset = offset
                                corrected_x_velocity = track[4] - pixel_velocities[0][0]
                                corrected_y_velocity = track[5] - pixel_velocities[0][1]
                                closestTrackPair = [pose_predictions[0].keypoints[i],
                                                    [corrected_x_velocity, corrected_y_velocity],
                                                    track_xyxy.astype(int)]
                if len(closestTrackPair) > 0:
                    pose_tracks.append(closestTrackPair)
        if len(pose_tracks) < 3:
            pose_tracks = []

        players_data = []
        for player in pose_tracks:
            player_pose = player[0]
            """
            The players data list combines the data of each track and pose pair with the data from the ball's track 
            and adds it to a list. Each of these will be input into a neural network 
            (if that is the classification option chosen)
            """
            players_data.append(np.concatenate([player[0].flatten().cpu().numpy(), np.array(player[1]),ball_data]))
            if chosen_classification_method == "basic":
                predict_reversed, predict_not_reversed = basic_classify(player_pose,player[1],predict_reversed,predict_not_reversed)


        if len(players_data) > 1 and chosen_classification_method!="basic":
            NN_input = torch.tensor(norm_scale_pipeline.transform(players_data),
                                    dtype=torch.float32) #preprocesses input data
            NN_input = NN_input.to(device) #sends the input tensor to GPU
            output = classify_reverse(NN_input) #gets output from neural network
            NN_input.to("cpu")
            output = output.cpu()
            output = output.detach().numpy().tolist()
            for i in range(len(output)):
                player_NN_output = output[i][0]

                if player_NN_output > 0.5: #if predicted reversed, then calculate how confident it is and then multiply by velocity
                    predict_reversed += (player_NN_output - 0.5) * (players_data[i][-5]**2 + players_data[i][-6]**2)**0.5
                else: #if predicted not reversed, then calculate how confident it is and then multiply by velocity
                    predict_not_reversed += (0.5 - player_NN_output) * (players_data[i][-5]**2 + players_data[i][-6]**2)**0.5

    if predict_not_reversed + predict_reversed > 0: #if any pose/track pairs were able to be processed
        reversibility_score = predict_reversed / (predict_not_reversed + predict_reversed)
    else:
        reversibility_score = 0
    reversibility_score = round(reversibility_score,3)*10
    if reversibility_score >= 5:
        if reversed_actual:
            true_positives.append([video_path,reversibility_score])
        else:
            false_positives.append([video_path,reversibility_score])
    else:
        if reversed_actual:
            false_negatives.append([video_path,reversibility_score])
        else:
            true_negatives.append([video_path,reversibility_score])

classification_accuracy = (len(true_positives) + len(true_negatives)) / (n_videos)
print ("Classification accuracy: ", classification_accuracy)

print ("True Positives: ",true_positives)
print ("True Negatives: ",true_negatives)
print ("False Positives: ",false_positives)
print ("False Negatives: ",false_negatives)









