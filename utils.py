import socket
import cv2
import dlib
import numpy as np
from aiohttp import web
import socketio
import base64
import io

def get_reference_point(cap, poster, fps, sio):

	# Shape of the poster
	pos_h, pos_w, _ = poster.shape
	ratio = pos_h/pos_w

	# Initializing to print the frame number
	framenum = 1

	# Initializing Face Detector
	detector = dlib.get_frontal_face_detector()

	# Setting up the switch: Switch is disabled as soon as a face is detected
	switch = True

	# Getting the height and width of the video frame
	fr_h, fr_w, _ = cap.read()[1].shape

	# Getting the center coordinates of the frame to check the placement of the poster
	fr_xc, fr_yc = fr_w//2, fr_h//2

	while True:
		# Reading the frames
		ret, frame = cap.read()

		# Break condition at the end of the video
		if not ret:
			break

		# Getting the faces in the frame
		faces = detector(frame)

		# If there is no any face and switch is True, then do nothing, just continue the loop
		if (not faces) and switch:
			cnt = cv2.imencode('.jpg',frame)[1]
            newFrame = base64.b64encode(cnt).decode('utf-8')
            await sio.emit('reply', newFrame, namespace='/get')
			print(f"Frame {framenum}: No Face Detected!")
			framenum += 1

			key = cv2.waitKey(1)
			if key == ord('q'):
				cv2.destroyAllWindows()
				break
				cap.release()

			continue

		# If there is a face and switch is True, get the top-left and bottom-right coordinates of the face and make switch False
		# Note: We define the injecting area as per the location of the face
		if faces and switch:

			# Looping over the faces in the frame
			for face in faces:

				# Switch is False as the face is detected
				switch = False

				# Getting the x1,y1 and x2,y2 coordinates of the face detected
				x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()

				# Getting the center coordinates of the face rectangle
				fc_xc, fc_yc = (x1 + x2)//2, (y1 + y2)//2

				#If the face-center is less than frame center (in the left side), then poster is injected on the right. And vice versa
				if fc_xc < fr_xc:
					print('Face on left and poster on right')
					top, right = int(fr_h * 0.05), int(fr_w * 0.92)

					if (fr_h >= fr_w) and (pos_h >= pos_w):
						print("Frame and poster both are potrait")
						left = right - (fr_w//4)
						bottom = top + int((right-left) * ratio)

					elif (fr_h >= fr_w) and (pos_h < pos_w):
						print("Frame is potrait and poster is landscape")
						bottom = top + (fr_w//4)
						left = right - int((bottom - top) / ratio)

					elif (fr_h < fr_w) and (pos_h >= pos_w):
						print("Frame is landscape and poster is potrait")
						left = right - (fr_h//4) 
						bottom = top + int((right - left) * ratio)

					else:
						print("Frame and poster both are landscape")
						bottom = top + (fr_h//4)
						left = right - int((bottom - top) / ratio)

					reference_point = top, right

				else:
					print('Face on right and poster on left')
					top, left = int(fr_h * 0.05), int(fr_w * 0.08)

					if (fr_h >= fr_w) and (pos_h >= pos_w):
						print("Frame and poster both are potrait")
						right = left + (fr_w//4)
						bottom = top + int((right - left) * ratio)

					elif (fr_h >= fr_w) and (pos_h < pos_w):
						print("Frame is potrait and poster is landscape")
						bottom = top + (fr_w//4)
						right = left + int((bottom - top) / ratio) 

					elif (fr_h < fr_w) and (pos_h >= pos_w):
						print("Frame is landscape and poster is potrait")
						right = left + (fr_h//4) 
						bottom = top + int((right - left) * ratio)

					else:
						print("Frame and poster both are landscape")
						bottom = top + (fr_h//4)
						right = left + int((bottom - top) / ratio)

					reference_point = top, left
		else:
			break

	return reference_point, left, right, top, bottom, framenum

#Function to get the upper and lower hsv range of values
def get_hsv_range(hsv, reference_point):

	threshold = 40

	hsvPoint = hsv[reference_point]

	lower = np.array([0, hsvPoint[1] - threshold, hsvPoint[2] - threshold])
	upper = np.array([255, hsvPoint[1] + threshold, hsvPoint[2] + threshold])

	return lower, upper