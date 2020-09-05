
import cv2 
 
cap = cv2.VideoCapture(0) 

if (cap.isOpened() == False): 
    print("Unable to read camera feed")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height),0)

while(cap.isOpened()): 
    
    ret, frame = cap.read() 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # draw line
    start_point = (0, 0)
    end_point = (250, 250)
    color = (0, 255, 0)
    thickness = 5
    cv2.line(img=gray, pt1=start_point, pt2=end_point, color=color, thickness=thickness, lineType=8, shift=0)
    
    # draw rectangle
    x1,y1 = 200, 200
    x2,y2 = 250, 250  
    cv2.rectangle(gray,(x1, y1), (x2, y2),color, 2)
    
    cv2.imshow('webcam(1)',gray)

    out.write(gray)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
  

cap.release() 
out.release()
cv2.destroyAllWindows() 