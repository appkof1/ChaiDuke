import cv2
import os

file_dir = "football_video_folder"
original_files = os.listdir(file_dir)
for file_name in original_files:
    cap = cv2.VideoCapture(f"{file_dir}/{file_name}")
    print (f"{file_dir}/{file_name}")
    test_file = cv2.VideoWriter(f'{file_dir}/reversed_{file_name}', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                    cap.get(cv2.CAP_PROP_FPS), (1280, 720))
    ret, frame = cap.read()
    frames = [frame]
    while ret:
        ret, frame = cap.read()
        frames.append(frame)
    frames.reverse()
    for frame in frames:
        test_file.write(frame)
    test_file.release()


