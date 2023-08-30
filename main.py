import pyfirmata
from time import sleep

import mediapipe as mp
import cv2
import math
import numpy as np
import os
import re


def variation(before:int, after:list):
    differences = []
    for i in range(len(after)-1):
        diff = after[i + 1] - after[i]
        differences.append(diff)
    if differences:
        after_mean = sum(differences) / len(differences)
        difference_x = before - after_mean
        # 動かした距離で判定
        if -50 > difference_x:
            return 1
    return None



def main():
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    # Webカメラから入力
    cap = cv2.VideoCapture(0)

    # データを見るパス
    path = './data/'
    # 端子を見分けるための正規表現
    # TODO:時間があればスマポ写真も
    patternStr = '.+\.(jpg|jpeg|png|bmp|gif|tiff|tif)'
    pattern = re.compile(patternStr)
    
    # LEDの設定
    PINS = ['d:2:o','d:3:o','d:4:o','d:5:o','d:6:o','d:7:o','d:8:o','d:9:o','d:10:o','d:11:o']
    PINNAMES = ['pin1','pin2','pin3','pin4','pin5','pin6','pin7','pin8','pin9','pin10']

    b=pyfirmata.Arduino('/dev/cu.usbmodem201912341')
    i = pyfirmata.util.Iterator(b)
    pin_objects = {}
    for pinname, pin in zip(PINNAMES, PINS):
        pin_objects[pinname] = b.get_pin(pin)
    i.start()

    
    before_INDEX_FINGER_TIP_x = 0
    INDEX_FINGER_TIP_x = []
    # 回数を数えるカウンター
    counter = 0
    # 動きをためておく変数
    stock = 0
    # 写真のカウンター
    phot_num = 0
    # onofスイッチ
    switch = 0
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        
        while cap.isOpened():
            # 画像のパスを取り出す
            if (counter == 0) or (counter >= 100):
                path_list = []
                for item in os.listdir(path):
                    match_object = pattern.match(item)
                    if match_object:
                        true_path = match_object.group()
                        path_list.append(true_path)
                counter = 1
            
            
            success, image = cap.read()
            h, w, _ = image.shape
            
            if not success:
                print("Ignoring empty camera frame.")
                continue
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            # 検出された手の骨格をカメラ画像に重ねて描画
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

                    # 座標を貯める
                    index_finger_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    INDEX_FINGER_TIP_x.append(int(index_finger_landmark.x * w))
                    
                    if stock > 7:
                        # 写真を他のにする処理
                        # 人差し指
                        num = variation(before_INDEX_FINGER_TIP_x,INDEX_FINGER_TIP_x)
                        
                        if num != None:
                            if len(path_list) > phot_num + num:
                                phot_num += num
                            else:
                                phot_num = 0

                        # リセット
                        INDEX_FINGER_TIP_x = []
                        stock = 0

            # LED削除
            if (phot_num == 0) and (switch == 1):
                for pinname in PINNAMES:
                    pin = pin_objects[pinname]
                    pin.write(0)
                switch=0
            # 見ている写真の場所を教えるLEDの数を変える
            if 0 <= phot_num <= (len(path_list)-1)*0.1:
                pin = pin_objects[PINNAMES[0]]
                pin.write(1)
            elif (len(path_list)-1)*0.1 < phot_num <= (len(path_list)-1)*0.2:
                pin = pin_objects[PINNAMES[1]]
                pin.write(1)
                switch = 1
            elif (len(path_list)-1)*0.2 < phot_num <= (len(path_list)-1)*0.3:
                pin = pin_objects[PINNAMES[2]]
                pin.write(1)
            elif (len(path_list)-1)*0.3 < phot_num <= (len(path_list)-1)*0.4:
                pin = pin_objects[PINNAMES[3]]
                pin.write(1)
            elif (len(path_list)-1)*0.4 < phot_num <= (len(path_list)-1)*0.5:
                pin = pin_objects[PINNAMES[4]]
                pin.write(1)
            elif (len(path_list)-1)*0.5 < phot_num <= (len(path_list)-1)*0.6:
                pin = pin_objects[PINNAMES[5]]
                pin.write(1)
            elif (len(path_list)-1)*0.6 < phot_num <= (len(path_list)-1)*0.7:
                pin = pin_objects[PINNAMES[6]]
                pin.write(1)
            elif (len(path_list)-1)*0.7 < phot_num <= (len(path_list)-1)*0.8:
                pin = pin_objects[PINNAMES[7]]
                pin.write(1)
            elif (len(path_list)-1)*0.8 < phot_num <= (len(path_list)-1)*0.9:
                pin = pin_objects[PINNAMES[8]]
                pin.write(1)
            elif (len(path_list)-1)*0.9 < phot_num <= (len(path_list)-1):
                pin = pin_objects[PINNAMES[9]]
                pin.write(1)

            # 画面表示設定
            height = h//2
            # 表示させる画像
            image_add = cv2.imread(os.path.join(path, path_list[phot_num]))
            h2, w2, _ =image_add.shape
            width = (height * w2)//(h2)
            size=(width,height)
            img_inter_area  = cv2.resize(image_add,size,interpolation = cv2.INTER_AREA)
            
            # 上半分を黒く
            image[:height, :] = [0, 0, 0]
            # 左上に画像を追加
            mid_w = (w - width)//2
            image[0:height, mid_w:mid_w + width] = img_inter_area
            
            image = cv2.flip(image, 1)
            image = cv2.arrowedLine(image, (mid_w + width, height + (height//2)), (mid_w, height + (height//2)), (255,99,71), thickness = 5, line_type=cv2.LINE_AA)
            cv2.imshow('MediaPipe Hands', image)
            
            if cv2.waitKey(5) & 0xFF == 27:
                break
            
            counter += 1
            stock += 1
    cap.release()

if __name__ == '__main__':
    main()