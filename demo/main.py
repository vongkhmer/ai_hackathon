from sneezing_firewall import SneezingFirewall
from video_streamer import VideoStreamer
from image_capturer import ImageCapturer
from sneezing_detector import SneezingDetector 
import cv2

def main():
	cap = ImageCapturer()
	cap = cv2.VideoCapture(1)
	sneezing_detector = SneezingDetector()
	sneezing_firewall = SneezingFirewall(cap, sneezing_detector)
	sneezing_firewall.start()
	video_streamer = VideoStreamer(0)
	while True:
		img = sneezing_firewall.get_img()
		k = video_streamer.stream(img)
		if k == 27:
			sneezing_firewall.terminate()
			break
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()