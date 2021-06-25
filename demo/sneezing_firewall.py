from threading import Thread
import numpy as np
import random
import cv2
import time

class SneezingFirewall(Thread):
	def __init__(self, image_capturer, sneezing_detector):
		Thread.__init__(self)	
		self.image_capturer = image_capturer
		#try reading one frame
		self.image = np.zeros((640, 480,3))
		while True:
			ret, frame = self.image_capturer.read()
			if ret > 0:
				self.image = frame.copy()
				break
		self.kill_state = False 
		self.sneeze_detector = sneezing_detector
		self.is_sneezing = 0
		self.sneezing_frame = 30
		self.count_down = 0
		self.dummy_image = cv2.resize(cv2.imread("resource/replacement.png"), (self.image.shape[1], self.image.shape[0]))

	def update_img(self):
		while not self.kill_state:
			ret, frame = self.image_capturer.read()
			if ret == 0:
				self.terminate()
				return
			self.image = frame.copy()
			# time.sleep(0.3)

	def sneeze_monitor(self):
		while not self.kill_state:
			self.is_sneezing = self.sneeze_detector.infer(self.image.copy())
			if self.is_sneezing:
				if self.count_down == 0:
					self.count_down = self.sneezing_frame 
			print(f"----------------------sneezing {self.is_sneezing}")
		# time.sleep(0.5)

	def run(self):
		thread1 = Thread(target=self.update_img)
		thread1.start()
		thread2 = Thread(target=self.sneeze_monitor)
		thread2.start()

	def get_img(self):
		if self.count_down > 0:
			self.count_down -= 1
			return self.dummy_image
		return self.image
	
	def terminate(self):
		self.kill_state = True
		self.image_capturer.release()
		print("Sneezing firewall is terminated")
