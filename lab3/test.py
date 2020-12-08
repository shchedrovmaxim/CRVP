import cv2
import numpy as np

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture("lab3/video_2020-11-01_13-52-43.mp4")

if (cap.isOpened()== False): 
  print("Error opening video stream or file")



width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter('test_video.avi', cv2.VideoWriter_fourcc(*'DIVX'),20,(width, height), isColor=True)

while(cap.isOpened()):

  ret, frame = cap.read()
  if ret == True:

  
    out.write(frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

  # Break the loop
  else: 
    break

# When everything done, release the video capture object
cap.release()
out.release()
# Closes all the frames
cv2.destroyAllWindows()
