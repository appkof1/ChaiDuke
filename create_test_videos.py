import cv2
import os
file_name = "city_liverpool.mp4" #videos are not included in submission as too large
file_name_2 = "brighton_liverpool.mp4"
new_folder_name = "football_video_folder"
cap = cv2.VideoCapture(file_name)
cap.set(cv2.CAP_PROP_POS_FRAMES, 30*500)

cap2 = cv2.VideoCapture(file_name_2)
cap2.set(cv2.CAP_PROP_POS_FRAMES, 30*500)
if not os.path.exists(new_folder_name):
    os.mkdir(new_folder_name)

seconds_per_video = 3
FPS = 30
frames_per_video = FPS * seconds_per_video
count = 0

for i in range(50):
    test_file = cv2.VideoWriter(f'{new_folder_name}/test_{count}.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), cap.get(cv2.CAP_PROP_FPS), (1280, 720))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 30*30*i+(30*60*10))
    for j in range(frames_per_video):
        ret, frame = cap.read()
        test_file.write(frame)
    test_file.release()
    print (f"test_{count}")
    count = count + 1

for i in range(50):
    test_file = cv2.VideoWriter(f'{new_folder_name}/test_{count}.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), cap.get(cv2.CAP_PROP_FPS), (1280, 720))
    cap2.set(cv2.CAP_PROP_POS_FRAMES, 30*30*i+(30*60*10))
    for j in range(frames_per_video):
        ret, frame = cap2.read()
        test_file.write(frame)
    test_file.release()
    print (f"test_{count}")
    count = count + 1

