from cv2 import cv2
import dlib
import numpy as np

#Function to get the upper and lower hsv range of values
def get_hsv_range(hsv, reference_point):

	threshold = 35

	hsvPoint = hsv[reference_point]

	lower = np.array([0, hsvPoint[1] - threshold, hsvPoint[2] - threshold])
	upper = np.array([255, hsvPoint[1] + threshold, hsvPoint[2] + threshold])

	return lower, upper

# Poster to inject
poster = cv2.imread("Posters/food panda_test.png")
pos_h, pos_w, _ = poster.shape
ratio = pos_h/pos_w

# Capturing the video frames
cap = cv2.VideoCapture('Videos/test3.mp4')
# cap = cv2.VideoCapture(0)

# Getting the height and width of the video frame
fr_h, fr_w, _ = cap.read()[1].shape

# Getting the center coordinates of the frame to check the placement of the poster
fr_xc, fr_yc = fr_w//2, fr_h//2

# Setting up the switch: Switch is disabled as soon as a face is detected
switch = True

# Initializing to print the frame number
framenum = 1

# Initializing Face Detector
detector = dlib.get_frontal_face_detector()

switcher = True

# Looping over all the video frames
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
		cv2.imshow('frame', frame)
		print(f"Frame {framenum}: No Face Detected!")
		framenum += 1

		key = cv2.waitKey(20)
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

				reference_point = right, top

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

				reference_point = left, top

	# Extracting the target-area from the frame to inject the poster
	area = frame[top:bottom, left:right, :]

	# Resizing the poster as per the defined area above. Note: cv2.resize() takes in as (w,h)
	poster = cv2.resize(poster, (right - left, bottom - top))

	# Convert the whole frame into HSV COLORSPACE
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	#Getting the lower and upper values
	lower, upper = get_hsv_range(hsv, reference_point)

	# Create kernel for image dilation
	kernel = np.ones((1,1), np.uint8)

	# Creating the mask using for the HSV frame using the upper and lower values
	mask = cv2.inRange(hsv, lower, upper)

	# Perform dilation on the mask to reduce noise
	dil = cv2.dilate(mask, kernel, iterations=5)

	# Now extract the area from the HSV frame with individual 3 channels
	mini_dil = np.zeros_like(area)
	mini_dil[:, :, 0] = dil[top: bottom, left: right]
	mini_dil[:, :, 1] = dil[top: bottom, left: right]
	mini_dil[:, :, 2] = dil[top: bottom, left: right]

	# Create the copy of the poster
	poster_copy = poster.copy()

	# Set pixel values of the poster_coy to 1 where pixel value of the mask is 0
	poster_copy[mini_dil == 0] = 1

	# Now set the pixel values in the target area to 1 where the pixel values of the poster_copy is not 1
	area[poster_copy != 1] = 1

	# Merge the poster_copy and the target area
	area = area * poster_copy

	# Now insert the final poster into the main frame
	frame[top: bottom, left:right, :] = area

	# Showing the final frame 
	cv2.imshow('frame', frame)
	print(f"Frame {framenum}: Done!")
	framenum += 1

	# Waiting for the key to be pressed by the user. If key is 'q' then quit everything
	key = cv2.waitKey(20)
	if key == ord('q'):
		cv2.destroyAllWindows()
		break
		cap.release()