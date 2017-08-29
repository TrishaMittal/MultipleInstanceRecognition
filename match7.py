import numpy as np
import argparse
import imutils
import glob
import cv2
import timeit

start = timeit.default_timer()

FLANN_INDEX_KDTREE = 1 
FLANN_INDEX_LSH    = 6
MAX_MIS_PREDICTIONS = 3

def filter_matches(kp1, kp2, matches, ratio = 0.8):
    mkp1, mkp2 = [], []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append( kp1[m.queryIdx] )
            mkp2.append( kp2[m.trainIdx] )
    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])
    kp_pairs = zip(mkp1, mkp2)
    return p1, p2, kp_pairs

def find_obj(img1, img2):
    detector = cv2.SIFT()
    norm = cv2.NORM_L2
    flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    matcher = cv2.FlannBasedMatcher(flann_params, {})
    kp1, desc1 = detector.detectAndCompute(img1, None)
    kp2, desc2 = detector.detectAndCompute(img2, None)
    #print 'img1 - %d features, img2 - %d features' % (len(kp1), len(kp2))
    if len(kp2)*10 < len(kp1) or len(kp1) < 2 or len(kp2) < 2:
	return False
    raw_matches = matcher.knnMatch(desc1, trainDescriptors = desc2, k = 2) #2
    p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches)
    #print 'matching...', len(p1)
    if len(p1) >= 4:
    	H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
	status_sum = np.sum(status)
	#print '%d / %d  inliers/matched' % (status_sum, len(status))
	#print 'Ratio : %d' % (len(kp1) / status_sum)
	if len(kp1) / status_sum <= 30:
        	return True
    return False



target = open('LIST1', 'r')
aa = target.readline()
counter = 0
while(aa[0]!='\n'):
	if (aa=="S\n"):
		a = target.readline().split(' ')
		image = cv2.imread("Shelf/"+a[0])
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		for i in range (int(a[1])):
			b = target.readline().split(' ')
			tpl = cv2.imread("DataBase/"+b[0])
			c = 0
			template = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)
			template = cv2.GaussianBlur(template,(5,5),0)
			print "------------------------------------"
			print 'SHELF = '+ a[0]+ ' TEMPLATE = '+ b[0]
			(tH, tW) = template.shape[:2]

			scale_space = []
			scales = []
			for scale in np.linspace(0.2, 1.0, 20)[::-1]:
				resized = imutils.resize(gray, width = int(gray.shape[1] * scale))
				if resized.shape[0] < tH or resized.shape[1] < tW:
					break
				r = gray.shape[1] / float(resized.shape[1])
				edged  = cv2.GaussianBlur(resized,(5,5),0)
				scale_space.append(cv2.matchTemplate(edged, template, cv2.TM_CCOEFF_NORMED))
				scales.append(r)

			count = MAX_MIS_PREDICTIONS
			scale_estimate = 0.0
			lower_scale = 0
			
			upper_scale = len(scales)
			while(count):
				#print "------------------"
				found = None
				for ind in range(lower_scale, upper_scale):
					(_, maxVal, _, maxLoc) = cv2.minMaxLoc(scale_space[ind])
					if found is None or maxVal > found[0]:
						found = (maxVal, maxLoc, ind)
				if found is not None:
					(maxVal, maxLoc, ind) = found
					r = scales[ind]
					#print 'maxLoc = ', found[0], r, maxLoc
					(startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
					(endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
					#print "Detected at: " ,maxLoc, "  score: ", maxVal," scale: ", r
					g = gray[startY:endY, startX:endX]
					rsz = cv2.resize(template, ( g.shape[1], g.shape[0]))
					g = cv2.GaussianBlur(g,(3,3),0)
					rsz = cv2.GaussianBlur(rsz,(3,3),0)
					#cv2.imwrite("op1.png", g);
					#cv2.imwrite("op2.png", rsz);
					flag = find_obj(rsz, g)
					if flag:
						cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 10)
						#cv2.imwrite("image.png", image)
						count = MAX_MIS_PREDICTIONS	
						c+=1
						# lower_scale = max(0, ind - 6);
						# upper_scale = min(len(scales), ind + 7)
						for k in range(len(scales)):
							startX = max(0, int(maxLoc[0]*scales[ind]/scales[k] - tW*scales[k]/4))
							startY = max(0, int(maxLoc[1]*scales[ind]/scales[k] - tH*scales[k]/4))
							# print '---', k, int(maxLoc[1]*scales[ind]/scales[k]), int(maxLoc[0]*scales[ind]/scales[k]), scale_space[k].shape
							#print startX, endX, startY, endY
							endX = min(int(maxLoc[0]*scales[ind]/scales[k] + tW*scales[k]/4), scale_space[k].shape[1])
							endY = min(int(maxLoc[1]*scales[ind]/scales[k] + tH*scales[k]/4), scale_space[k].shape[0])
							for i in range(startX, endX):
								for j in range(startY,endY):
									scale_space[k][j][i] = 0.0
						
					else:
						count -= 1

				else:
					count -= 1
			print "DETECTIONS = " + str(c),
			print " GROUND TRUTH = " + b[1]
			cv2.imwrite("Auto_Results1/res" + str(counter) + ".png", image)
			# Create a black image, a window and bind the function to window
   			img = np.zeros((512,512,3), np.uint8)
   			#ssssscv2.setMouseCallback('mouse input',onmouse)

			counter += 1
	aa = target.readline()
stop = timeit.default_timer()

print stop - start 
