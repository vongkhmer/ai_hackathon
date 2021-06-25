import cv2
import random

class ImageCapturer:
	def __init__(self):
		pass

	def read(self):
		imgn = ["1.png", "2.jpg", "3.jpg"]
		id = random.randint(0, len(imgn) - 1)
		print(f"got {id} -> {imgn[id]}")
		ret = 1
		frame = cv2.imread("resource/" + imgn[id])
		return ret, frame

	def release(self):
		pass