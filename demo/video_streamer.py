import cv2

class VideoStreamer:
	def __init__(self, destination):
		self.destination = destination

	def stream(self, img):
		cv2.imshow("test", img)
		return cv2.waitKey(1)