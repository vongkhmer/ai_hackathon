from models import *
from openvino.inference_engine import IECore
import cv2

class SneezingDetector:
	def __init__(self):
		ie = IECore()
		facedetector_model_xml = "/Users/vongkh/Documents/Programming/AI/face-landmark/intel/face-detection-adas-0001/FP32/face-detection-adas-0001.xml"
		facedetector_model_bin = "/Users/vongkh/Documents/Programming/AI/face-landmark/intel/face-detection-adas-0001/FP32/face-detection-adas-0001.bin"
		landmark_model_xml = "/Users/vongkh/intel/facial-landmarks-35-adas-0002/FP32/facial-landmarks-35-adas-0002.xml"
		landmark_model_bin = "/Users/vongkh/intel/facial-landmarks-35-adas-0002/FP32/facial-landmarks-35-adas-0002.bin"
		self.land_mark_model = IEModel(landmark_model_xml, landmark_model_bin, ie, target_device='CPU', num_requests=1)
		self.facedetector_model = IEModel(facedetector_model_xml, facedetector_model_bin, ie, target_device='CPU', num_requests=1)

		self.landmark_input_size = self.land_mark_model.input_size
		self.facedetector_input_size = self.facedetector_model.input_size
		self.sample_num = 0
		self.mean = 0.4
		self.square_mean = 0.16
		self.mouth_open = 0
	
	def infer(self, frame):
		frame = cv2.resize(frame, (self.facedetector_input_size[-1], self.facedetector_input_size[-2]))
		print(frame.shape)
		facedetector_input = frame.transpose((2,0,1))
		print(facedetector_input.shape)
		facedetector_input = np.expand_dims(facedetector_input, axis = 0)
		res = self.facedetector_model.infer(facedetector_input)
		bb = get_bounding_box(frame, res[0])
		if len(bb) == 0:
			return 0

		(x_min, y_min, x_max, y_max) = bb[0]
		w = x_max - x_min
		h = y_max - y_min
		# print("after adjust", x_min, y_min, w, h)
		frame = crop_img(frame, x_min, y_min, w, h) 
		# print("frame", frame.shape)
		
		landmarkdetector_input = cv2.resize(frame, (self.landmark_input_size[-1], self.landmark_input_size[-1])) 
		landmarkdetector_input = landmarkdetector_input.transpose((2, 0, 1))
		landmarkdetector_input = np.expand_dims(landmarkdetector_input, axis = 0)

		# print("land input", landmarkdetector_input.shape)

		res = self.land_mark_model.infer(landmarkdetector_input)
		# print("res shape", res.shape)

		frame = preprocess_frame(frame, size=400) 

		landmark = convert_landmark_coordinate(res[0], x_min, y_min, w, h)

		mw, mh = get_mouth_size(landmark)
		ms = mh * mw
		
		if self.sample_num < 200:
			self.mean = (self.mean * self.sample_num + ms) / (self.sample_num + 1)
			self.square_mean = (self.square_mean * self.sample_num + ms * ms) / (self.sample_num + 1)
			self.sample_num += 1
			self.std = math.sqrt(self.square_mean - self.mean * self.mean)
			print("calibrating...")
			return 0
		else:
			if ms - self.mean > 3 * self.std:
				self.mouth_open += 1
			else:
				self.mouth_open = 0
			
			if self.mouth_open > 5:
				return 1
			return 0