import sys

import numpy as np
from PIL import Image

sys.path.insert(0, 'face_detection/mtcnn/')
from src.detector_class import Detector
from arcface_warping import preprocess as warping

sys.path.insert(0, 'face_recognition')
from face_recognition import FaceRecognizer

def main():

	img = Image.open('data/example_img.jpg')
	face_detector = Detector(weights_prefix_path='face_detection/mtcnn/')
	face_recognizer = FaceRecognizer(weights='face_recognition/weights/face_recognition_backbone_ir50_ms1m_epoch120.pth')

	# Detection of all faces in the image. 
	# For large images use pre-resize (for example, to 640px for the larger dimension).
	bboxes, landmarks = face_detector.detect_faces(img)

	descriptors = []
	for idx, (bbox, landmarks5) in enumerate(zip(bboxes, landmarks)):
		#Face alignment by 5 landmark points
		warped_img = warping(np.array(img), landmarks5.reshape((2,5)).T)
		#Convert from numpy array to PIL Image
		warped_img = Image.fromarray(warped_img)
		warped_img.save(f'data/face_{idx}.jpg')
		descriptor = face_recognizer.get_descriptor(warped_img)
		descriptors.append(descriptor)

	if len(descriptors) > 1:
		for idx in range(len(descriptors[:-1])):
			for jdx in range(idx + 1, len(descriptors)):
				descriptor_a = descriptors[idx]
				descriptor_b = descriptors[jdx]
				distance = np.linalg.norm(descriptor_a - descriptor_b)
				# distance < 1.0 - the same person.
				print(f'Distance between Face {idx} and Face {jdx}: {distance:.4f}')

if __name__ == '__main__':
	main()





