import math
from collections import deque
from itertools import cycle

import cv2
import numpy as np

def adaptive_resize(frame, dst_size):
    h, w, _ = frame.shape
    scale = dst_size / max(h, w)
    ow, oh = int(w * scale), int(h * scale)

    if ow == w and oh == h:
        return frame
    return cv2.resize(frame, (ow, oh))

def preprocess_frame(frame, size=224):
    frame = adaptive_resize(frame, size)
    # frame = frame.transpose((2, 0, 1))  # HWC -> CHW
    return frame

def convert_landmark_coordinate(landmark, x_min, y_min, w, h):
	landmark_num = len(landmark) // 2
	res = []
	for i in range(landmark_num):
		x, y = landmark[2 *i : 2 * i + 2]
		x = math.ceil(x * w) 
		y = math.ceil(y * h)
		x += x_min
		y += y_min
		res.extend([x,y])
	return res	

def adjust_bounding_box(bb):
	x_min, y_min, x_max, y_max = bb
	w = x_max - x_min
	h = y_max - y_min
	x_min = int(round(x_min - w * 0.067))
	y_min = int(round(y_min - h * 0.028))

	w = int(round(w + 0.15 * w))
	h = int(round(h + 0.13 * h))

	if w < h:
		dx = h - w
		w += dx
		x_min -= dx // 2
	else:
		dy = w -h
		h += dy
		y_min -= dy // 2
	x_max = x_min + w
	y_max = y_min + h
	return x_min, y_min, x_max, y_max 

def euclidean_dist(p1, p2):
	return math.sqrt((p1[0] -p2[0]) ** 2  + (p1[1] - p2[1]) ** 2)

def get_mouth_size(landmark):
	x10, y10 = landmark[2*10:2*10+2]
	x11, y11 = landmark[2*11:2*11+2]
	x8, y8 = landmark[2*8:2*8+2]
	x9, y9 = landmark[2*9:2*9+2]
	x6, y6 = landmark[2*6:2*6+2]
	x7, y7 = landmark[2*7:2*7+2]
	w = euclidean_dist((x8, y8), (x9, y9))
	h = euclidean_dist((x10, y10), (x11, y11))
	norm = euclidean_dist((x6,y6), (x7, y7))
	return w / norm, h/norm


def get_bounding_box(frame, faces):
	res = []
	h,w,_ = frame.shape
	faces = faces[0]
	for i in range(faces.shape[0]):
		_,_,p,x_min,y_min,x_max,y_max = faces[i,:]
		if p < 0.3:
			break
		# print(p)
		x_min = int(round(x_min * w))
		y_min = int(round(y_min * h))
		x_max = int(round(x_max * w))
		y_max = int(round(y_max * h))
		x_min, y_min, x_max, y_max = adjust_bounding_box((x_min, y_min, x_max, y_max))
		w = x_max - x_min
		h = y_max - y_min
		# print("after adjust", w, h)
		res.append((x_min, y_min, x_max, y_max))
		# frame = cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255,0), 3)
	return res


def crop_img(img, x0, y0, w, h):
	res = np.zeros((h,w, 3))
	x_start = max(x0, 0)
	y_start = max(y0, 0)
	x_end = min(x0 + w, img.shape[1])
	y_end = min(y0 + h, img.shape[0])
	res[y_start - y0:y_end - y0, x_start - x0:x_end - x0,:] = img[y_start:y_end, x_start:x_end,:]
	return res

def square_pad(img):
	return img

class IEModel:
    def __init__(self, model_xml, model_bin, ie_core, target_device, num_requests, batch_size=1):
        print("Reading IR...")
        self.net = ie_core.read_network(model_xml, model_bin)
        self.net.batch_size = batch_size
        assert len(self.net.input_info) == 1, "One input is expected"
        assert len(self.net.outputs) == 1, "One output is expected"

        print("Loading IR to the plugin...")
        self.exec_net = ie_core.load_network(network=self.net, device_name=target_device, num_requests=num_requests)
        self.input_name = next(iter(self.net.input_info))
        self.output_name = next(iter(self.net.outputs))
        self.input_size = self.net.input_info[self.input_name].input_data.shape
        self.output_size = self.exec_net.requests[0].output_blobs[self.output_name].buffer.shape
        self.num_requests = num_requests

    def infer(self, frame):
        input_data = {self.input_name: frame}
        infer_result = self.exec_net.infer(input_data)
        return infer_result[self.output_name]

    def async_infer(self, frame, req_id):
        input_data = {self.input_name: frame}
        self.exec_net.start_async(request_id=req_id, inputs=input_data)
        pass

    def wait_request(self, req_id):
        self.exec_net.requests[req_id].wait()
        return self.exec_net.requests[req_id].output_blobs[self.output_name].buffer