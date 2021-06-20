# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 15:04:46 2021

@author: kankan
"""

import threading
import keyboard


import cv2
import pyvirtualcam
import numpy as np
import time

from obswebsocket import obsws, requests


event = threading.Event()

KILLevent = threading.Event()




def Controller():
    event.set()  # 初期値はTrue
    
    KILLevent.clear()
    
    while True:
        if keyboard.read_key() == "p":
            event.clear()
            while True:
                if keyboard.read_key() == "q":
                    event.set()
                    break
        elif keyboard.read_key() == "b":
            KILLevent.set()
            break
    
    print("Controller is dead")
    

def OBS():
    #OBSのポート設定
    host = "localhost"
    port = 4444
    password = "1234"
    
    #OBSに接続
    ws = obsws(host, port, password)
    ws.connect()
    
    #OBSのソース名を取得
    source = ws.call(requests.GetSourcesList())
    if source.status:
        for s in source.getSources():
            print(s["name"])
            sourcename=s["name"]
            
            
            

    
    # 取得するデータソース（カメラ）を選択
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    
    # 最初のフレームから画像のサイズを取得
    ret, frame = cap.read()
    
    #塗りつぶし範囲を指定
    pts = np.array([[0,0],[0,frame.shape[0]],[frame.shape[1],frame.shape[0]],[frame.shape[1],0]])
    with pyvirtualcam.Camera(width=frame.shape[1], height=frame.shape[0], fps=30) as cam:
        while cap.isOpened():
            if not ret:
                break
            
            #終了信号
            if KILLevent.is_set():
                break
            
            
            
            #フラグチェック
            #画面を隠す処理
            if event.is_set():
                # 色空間を変更RGB順にする
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 画像を仮想カメラに流す
                cam.send(frame)
                
                # 画像をスクリーンに表示しなくなったので，pyvirtualcamの機能を使って次のフレームまで待機する
                cam.sleep_until_next_frame()        
                
                # 各フレームの画像を取得
                ret, frame = cap.read()
                    
            else:
                #ミュート
                ws.call(requests.SetMute(sourcename, True))
                #塗りつぶし
                cv2.fillPoly(frame,pts=[pts], color=(0,0,0))
                # 色空間を変更RGB順にする
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 画像を仮想カメラに流す
                cam.send(frame)
                
                # 画像をスクリーンに表示しなくなったので，pyvirtualcamの機能を使って次のフレームまで待機する
                cam.sleep_until_next_frame()   
                
                #フラグがTrueになるまで待機
                event.wait()
                
                #ミュート解除
                ws.call(requests.SetMute(sourcename, False))
                # 各フレームの画像を取得
                ret, frame = cap.read()


    # 終了処理
    time.sleep(1)
    cap.release()
    cv2.destroyAllWindows()
    ws.disconnect()
    
    print("obs_c is dead")
        

obs_c = threading.Thread(target=OBS,)
obs_c.start()

controller = threading.Thread(target=Controller,)
controller.start()

