import cv2
import numpy as np
import math
import telegram
import timeit
import time

count = 0


cap = cv2.VideoCapture(0)  ##
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 500)  ##
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500) ##

BODY_PARTS = { "Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
                "Background": 15 }

POSE_PAIRS = [ ["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
                ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
                ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
                ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"] ]

protoFile = "C:\\Users\\user\\Desktop\\python_data\\pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = "C:\\Users\\user\\Desktop\\python_data\\pose_iter_160000.caffemodel"

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile) # net, 네트워크 불러옴

nPoints = 15
POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14],
              [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]
inWidth = 368
inHeight = 368

threshold = 0.1

telgm_token = '2092441658:AAGQihaN3s94-UFNeTHmuoPpWFP2PDjYte4'

bot = telegram.Bot(token = telgm_token)
  



## 역탄젠트 구하기
def calculate_degree(point_1, point_2, frame):
    global count
    dx = point_2[0] - point_1[0]
    dy = point_2[1] - point_1[1]
    rad = math.atan2(abs(dy), abs(dx))

    # radian 을 degree 로 변환
    deg = rad * 180 / math.pi

    if deg < 45:
        count += 1
        string = "Fall Down"
        #cv2.putText(frame, string, (0, 25), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 255))
        print(f"[degree] {deg} ({string})")
        

        if count < 5:
            bot.sendMessage(chat_id = '2078372905', text="Fall Down.")
        elif count >= 5:
            bot.sendMessage(chat_id = '2078372905', text="Emergency.")
            
        
    else:
        string = "Stand"
        count = 0
        #cv2.putText(frame, string, (0, 25), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 255))
        print(f"[degree] {deg} ({string})")
        bot.sendMessage(chat_id = '2078372905', text="Stand.")



while cv2.waitKey(1) < 0:
    
    hasFrame, frame = cap.read()
    frameCopy = np.copy(frame)
    gray = cv2.cvtColor(frameCopy, cv2.COLOR_BGR2GRAY)
   
    
    if not hasFrame:
        
            cv2.waitKey()
            break
        
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), ## 네트워크에 넣기 위한 전처리
                                  (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inpBlob) # 전처리된 blob 네트워크에 입력
     
    # 결과 받아오기
    output = net.forward()

    # output.shape[0] = 이미지 ID, [1] = 출력 맵의 높이, [2] = 너비
    H = output.shape[2]
    W = output.shape[3]

    # 키포인트 검출시 이미지에 그려줌
    points = []
    
    for i in range(nPoints):
            # 해당 신체부위 신뢰도 얻음.
            probMap = output[0, i, :, :]

            # global 최대값 찾기
            # 최소값, 최대값, 최소값 위치, 최대값 위치
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
            
            # 원래 이미지에 맞게 점 위치 변경
            x = (frameWidth * point[0]) / W
            y = (frameHeight * point[1]) / H

            if prob > threshold : 
                cv2.circle(gray, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.putText(gray, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
                points.append((int(x), int(y)))
            else :
                points.append(None)
                
    frame_line = frame.copy()
    

        # Neck 과 MidHeap 의 좌표값이 존재한다면
    if (points[1] is not None) and (points[8] is not None):
       calculate_degree(point_1=points[1], point_2=points[8], frame=frame_line)
       
    for pair in POSE_PAIRS:
            partA = pair[0] # 0 (Head)
            partB = pair[1] # 1 (Neck)
            
            if points[partA] and points[partB]:
                cv2.line(gray, points[partA], points[partB], (0, 255, 255), 3, lineType=cv2.LINE_AA)

    
    cv2.imshow("VideoFrame", gray)
     
cap.release()
cv2.destroyAllWindows()


## 2092441658:AAGQihaN3s94-UFNeTHmuoPpWFP2PDjYte4
## chat id = 2078372905
