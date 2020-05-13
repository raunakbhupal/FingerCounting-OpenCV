import cv2
import numpy as np 
from sklearn.metrics import pairwise


background =None
acc_weight = 0.5
roi_top=20
roi_bottom=300
roi_right =300
roi_left=600


def cal_acc_weight(frame,acc_weight):
    global background
    if background is None:
        background=frame.copy().astype('float')
        return None
    cv2.accumulateWeighted(frame,background,acc_weight)
    

def segment(frame,threshold_min=25):
    diff=cv2.absdiff(background.astype('uint8'),frame)
    ret,threshold=cv2.threshold(diff,threshold_min,255,cv2.THRESH_BINARY)
    contours,hierarchy=cv2.findContours(threshold.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours)==0:
        return None
    else:
        hand_segment=max(contours,key=cv2.contourArea)
        return (threshold,hand_segment)
    

def count_fingers(thershold,hand_segment):
    conv_hull=cv2.convexHull(hand_segment)
    top=tuple(conv_hull[conv_hull[:,:,1].argmin()][0])
    bottom=tuple(conv_hull[conv_hull[:,:,1].argmin()][0])
    left=tuple(conv_hull[conv_hull[:,:,0].argmin()][0])
    right=tuple(conv_hull[conv_hull[:,:,0].argmin()][0])
    cX=(left[0]+right[0])//2
    cY=(top[1]+bottom[1])//2
    distance=pairwise.euclidean_distances([(cX,cY)],Y=[left,right,top,bottom])[0]
    max_distance=distance.max()
    radius=int(0.9*max_distance)
    circumference=(2*np.pi*radius)
    circular_roi = np.zeros(threshold.shape[:2], dtype="uint8")
    cv2.circle(circular_roi, (cX, cY), radius, 255, 10)
    circular_roi = cv2.bitwise_and(threshold, threshold, mask=circular_roi)
    contours, hierarchy = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    count = 0
    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        out_of_wrist = ((cY + (cY * 0.25)) > (y + h))
        limit_points = ((circumference * 0.25) > cnt.shape[0])
        if  out_of_wrist and limit_points:
            count += 1
    return count


cam=cv2.VideoCapture(0)
num_frames=0
while True:
    ret,frame=cam.read()
    frame = cv2.flip(frame, 1)
    frame_copy=frame.copy()
    roi=frame[roi_top:roi_bottom,roi_right:roi_left]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    if num_frames < 60:
        cal_acc_weight(gray, acc_weight)
        if num_frames <= 59:
            cv2.putText(frame_copy, "WAIT! GETTING BACKGROUND AVG.", (200,300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.imshow("Finger Count",frame_copy)
    else:
        hand=segment(gray)
        if hand is not None:
            threshold, hand_segment = hand
            cv2.drawContours(frame_copy, [hand_segment + (roi_right, roi_top)], -1, (255, 0, 0),1)
            fingers = count_fingers(threshold,hand_segment)
            cv2.putText(frame_copy, str(fingers), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.imshow("Thesholded", threshold)
    
    cv2.rectangle(frame_copy, (roi_left, roi_top), (roi_right, roi_bottom), (0,0,255), 5)
    num_frames += 1
    cv2.imshow("Finger Count", frame_copy)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
cam.release()
cv2.destroyAllWindows()