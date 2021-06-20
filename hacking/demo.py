import time
import cv2
import numpy as np
from openvino.inference_engine import IECore

print("reading label...")
label = open('labels.txt').readlines()

print(len(label), 'labels read')   # 読み込んだラベルの個数を表示
print(label[:20])  


def decode_output(probs, labels, top_k=None, label_postprocessing=None):
    """Decodes top probabilities into corresponding label names"""
    top_ind = np.argsort(probs)[::-1][:top_k]

    if label_postprocessing:
        for k in range(top_k):
            label_postprocessing[k].update(top_ind[k])

        top_ind = [postproc.get() for postproc in label_postprocessing]

    decoded_labels = [labels[i] if labels else str(i) for i in top_ind]
    probs = [probs[i] for i in top_ind]
    return decoded_labels, probs

def center_crop(frame, crop_size):
    img_h, img_w, _ = frame.shape

    x0 = int(round((img_w - crop_size[0]) / 2.))
    y0 = int(round((img_h - crop_size[1]) / 2.))
    x1 = x0 + crop_size[0]
    y1 = y0 + crop_size[1]

    return frame[y0:y1, x0:x1, ...]


def adaptive_resize(frame, dst_size):
    h, w, _ = frame.shape
    scale = dst_size / min(h, w)
    ow, oh = int(w * scale), int(h * scale)

    if ow == w and oh == h:
        return frame
    return cv2.resize(frame, (ow, oh))

def preprocess_frame(frame, size=224, crop_size=224):
    frame = adaptive_resize(frame, size)
    frame = center_crop(frame, (crop_size, crop_size))
    frame = frame.transpose((2, 0, 1))  # HWC -> CHW

    return frame

def softmax(x):
    u = np.sum(np.exp(x))
    ans = []
    for i in x:
        ans.append(np.exp(i) / u)
    return ans

# Inference Engineコアオブジェクトの生成
ie = IECore()

# IRモデルファイルの読み込み
en_net = ie.read_network(model="./models/action-recognition-0001-encoder.xml",  weights="./models/action-recognition-0001-encoder.bin")

# IRモデルファイルの読み込み
de_net = ie.read_network(model="./models/action-recognition-0001-decoder.xml",  weights="./models/action-recognition-0001-decoder.bin")

# 入出力blobの名前の取得、入力blobのシェイプの取得
en_input_blob_name  = list(en_net.input_info.keys())[0]
en_output_blob_name = list(en_net.outputs.keys())[0]
print(en_input_blob_name)

batch,channel,height,width =en_net.input_info[en_input_blob_name].tensor_desc.dims
print("batch size", batch)

de_input_blob_name  = list(de_net.input_info.keys())[0]
de_output_blob_name = list(de_net.outputs.keys())[0]
# print(en_net.input_info[en_input_blob_name].tensor_desc.dims)

exec_net = ie.load_network(network=en_net, device_name='CPU', num_requests=1)
exec_net2 = ie.load_network(network=de_net, device_name='CPU', num_requests=1)

cap = cv2.VideoCapture(1)
de_input_buff = []
NUM_FRAME = 16 
frame_NUM = 0
last_prediction = ""

def put_text(img, txt):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, txt, (10,450), font, 3, (0, 255, 0), 2, cv2.LINE_AA)

while(1):
    frame_NUM += 1
    ret, frame = cap.read()
    # frame = cv2.imread("sneezing4.jpeg")
    put_text(frame, last_prediction)
    cv2.imshow('Video', frame)
    img = np.expand_dims(preprocess_frame(frame), axis=0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    res = exec_net.infer(inputs={en_input_blob_name: img})
    de_input_buff.append(res[en_output_blob_name][0].reshape(512))
    if len(de_input_buff) < NUM_FRAME:
        continue
    if len(de_input_buff) > NUM_FRAME:
        de_input_buff.pop(0)

    de_input = np.stack(de_input_buff, axis=0).reshape(1, NUM_FRAME, 512)

    res2 = exec_net2.infer(inputs={de_input_blob_name: de_input})
    res2 = res2[de_output_blob_name][0]
    print(res2.shape)
				#確率が最も大きい要素の計算
    decoded_label, prob = decode_output(res2, label, top_k=5)
    print(decoded_label, prob)
    last_prediction = decoded_label[0]
    # probability = softmax(res2[de_output_blob_name].reshape((400)))
    # max_index = np.argmax(probability)
    # print(max_index + 1, probability[max_index])
cap.release()
cv2.destroyAllWindows()