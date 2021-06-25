from itertools import starmap
from models import *
from openvino.inference_engine import IECore
import math
import time

ie = IECore()
facedetector_model_xml = "/Users/vongkh/Documents/Programming/AI/face-landmark/intel/face-detection-adas-0001/FP32/face-detection-adas-0001.xml"
facedetector_model_bin = "/Users/vongkh/Documents/Programming/AI/face-landmark/intel/face-detection-adas-0001/FP32/face-detection-adas-0001.bin"
landmark_model_xml = "/Users/vongkh/intel/facial-landmarks-35-adas-0002/FP32/facial-landmarks-35-adas-0002.xml"
landmark_model_bin = "/Users/vongkh/intel/facial-landmarks-35-adas-0002/FP32/facial-landmarks-35-adas-0002.bin"
land_mark_model = IEModel(landmark_model_xml, landmark_model_bin, ie, target_device='CPU', num_requests=1)
facedetector_model = IEModel(facedetector_model_xml, facedetector_model_bin, ie, target_device='CPU', num_requests=1)

landmark_input_size = land_mark_model.input_size
facedetector_input_size = facedetector_model.input_size
print("input size", landmark_input_size)
print("input size", facedetector_input_size)

cap = cv2.VideoCapture(1)

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

def plot_landmark(img, landmark):
	landmark_num = len(landmark) // 2
	w, h, _= img.shape
	for i in range(landmark_num):
		x, y = landmark[2 *i : 2 * i + 2]
		img = cv2.circle(img, (x,y), 2, (255, 0, 0), -1)
	return img

def draw_faces(frame, faces):
	h,w,_ = frame.shape
	faces = faces[0]
	for i in range(faces.shape[0]):
		_,_,p,x_min,y_min,x_max,y_max = faces[i,:]
		if p < 0.3:
			break
		print(p)
		x_min = int(round(x_min * w))
		y_min = int(round(y_min * h))
		x_max = int(round(x_max * w))
		y_max = int(round(y_max * h))
		print(x_min, y_min, x_max, y_max)
		frame = cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255,0), 3)
	return frame

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
		print(p)
		x_min = int(round(x_min * w))
		y_min = int(round(y_min * h))
		x_max = int(round(x_max * w))
		y_max = int(round(y_max * h))
		x_min, y_min, x_max, y_max = adjust_bounding_box((x_min, y_min, x_max, y_max))
		w = x_max - x_min
		h = y_max - y_min
		print("after adjust", w, h)
		res.append((x_min, y_min, x_max, y_max))
		# frame = cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255,0), 3)
	return res

def puttxt(img, txt, pos, c = None):
	txt = str(txt)
	# font
	font = cv2.FONT_HERSHEY_SIMPLEX
		
	# org
		
	# fontScale
	fontScale = 0.5
		
	# Blue color in BGR
	if not c:
		color = (255, 0, 0)
	else:
		color = c
		
	# Line thickness of 2 px
	thickness = 2
		
	# Using cv2.putText() method
	image = cv2.putText(img, txt, pos, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
	return image

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

mean = 0.4
square_mean = 0.16
sample_num = 0
mouth_open = 0
frame_num = 0
start_time = time.time() * 1000

while(1):
	frame_num += 1
	ret, frame = cap.read()
	frame = cv2.resize(frame, (facedetector_input_size[-1], facedetector_input_size[-2]))
	original_frame = frame.copy()
	print(frame.shape)
	facedetector_input = frame.transpose((2,0,1))
	print(facedetector_input.shape)
	facedetector_input = np.expand_dims(facedetector_input, axis = 0)
	res = facedetector_model.infer(facedetector_input)
	# frame = draw_faces(frame, res[0])
	# # print(res.shape)
	bb = get_bounding_box(frame, res[0])
	if len(bb) == 0:
		cv2.imshow('Video', frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		continue
	original_frame = draw_faces(original_frame, res[0])

	(x_min, y_min, x_max, y_max) = bb[0]
	w = x_max - x_min
	h = y_max - y_min
	print("after adjust", x_min, y_min, w, h)
	frame = crop_img(frame, x_min, y_min, w, h) 
	print("frame", frame.shape)
	
	landmarkdetector_input = cv2.resize(frame, (landmark_input_size[-1], landmark_input_size[-1])) 
	landmarkdetector_input = landmarkdetector_input.transpose((2, 0, 1))
	landmarkdetector_input = np.expand_dims(landmarkdetector_input, axis = 0)

	print("land input", landmarkdetector_input.shape)

	res = land_mark_model.infer(landmarkdetector_input)
	print("res shape", res.shape)

	frame = preprocess_frame(frame, size=400) 

	landmark = convert_landmark_coordinate(res[0], x_min, y_min, w, h)

	frame = plot_landmark(original_frame, landmark)
	mw, mh = get_mouth_size(landmark)
	ms = mh * mw
	
	if sample_num < 200:
		mean = (mean * sample_num + ms) / (sample_num + 1)
		square_mean = (square_mean * sample_num + ms * ms) / (sample_num + 1)
		sample_num += 1
		std = math.sqrt(square_mean - mean * mean)
		frame = puttxt(frame, f"calibrating", (50, 75))
	else:
		frame = puttxt(frame, ms, (50, 50))
		frame = puttxt(frame, f"mean : {mean}", (50, 75))
		frame = puttxt(frame, f"std : {std}", (50, 100))
		prediction = "Normal"
		color = (255, 0, 0)
		if ms - mean > 3 * std:
			mouth_open += 1
		else:
			mouth_open = 0

		if mouth_open > 5:
			prediction = "Sneezing"
			color = (0,0,255)
		frame = puttxt(frame, f"State: {prediction}", (50, 125), color)


	cv2.imshow('Video', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
			break
    

	
cap.release()
cv2.destroyAllWindows()