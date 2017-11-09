import numpy as np
import argparse
import imutils
import glob
import cv2
import timeit

FLANN_INDEX_KDTREE = 1 
FLANN_INDEX_LSH    = 6
MAX_MIS_PREDICTIONS = 3
COLORS = [(60, 20, 220), (255, 255, 255), (98, 28, 139), (255, 0, 255), (255, 245, 0),  (0, 255, 127), (0, 255, 0)]
output = open('output8', 'w')
brands = {}

def get_template_descriptors():
	fp1 = open('files/l_tea','r')
	names = fp1.readlines()
	desc = []
	desc = np.asarray(desc)
	flag = False
	classes = {}
	count = 0
	c = 1
	tpl_names = {}
	for file_name in names:
		temp = file_name.split()
		file_name = temp[0]
		brands[temp[0]] = temp[1:]
		tpl_names[c] = file_name
		print file_name
		img = cv2.imread(file_name)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		b,g,r = cv2.split(img)
		kp1 = detector.detect(gray)
		kp1, desc1= detector.compute(b, kp1)
		kp1, desc_testg = detector.compute(g, kp1)
		desc1  = np.concatenate((desc1, desc_testg), axis = 1)
		kp1, desc_testr = detector.compute(r, kp1)
		desc1  = np.concatenate((desc1, desc_testr), axis = 1)
		#kp1, desc1 = detector.detectAndCompute(img, None)
		for i in range(len(kp1)):
			classes[count] = c
			count += 1
		c += 1
		if(flag):
			desc  = np.concatenate((desc, desc1), axis = 0) 
		else:
			flag = True
			desc = desc1
	fp1.close()
	return desc, classes, tpl_names

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


start = timeit.default_timer()
shelf = open('files/s_tea', 'r')
counter = 0
sift = cv2.SIFT() 
detector = cv2.SIFT()
norm = cv2.NORM_L2
matcher = cv2.BFMatcher(norm)
tpl_desc, classes, tpl_names = get_template_descriptors()

for sh in shelf.readlines():
	start1 = timeit.default_timer()
	c_count = 0
	print sh
	output.write('S '+ sh)
	image = cv2.imread(sh[:-1])
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	b,g,r = cv2.split(image)
	kp_test = detector.detect(gray)
	kp_test, desc_test = detector.compute(b, kp_test)
	kp_test, desc_testg = detector.compute(g, kp_test)
	desc_test  = np.concatenate((desc_test, desc_testg), axis = 1)
	kp_test, desc_testr = detector.compute(r, kp_test)
	desc_test  = np.concatenate((desc_test, desc_testr), axis = 1)
	raw_matches = matcher.knnMatch(tpl_desc, trainDescriptors = desc_test, k = 1) #2
	raw_matches = sorted(raw_matches, key = lambda x:x[0].distance)
	predict = {}
	for m in raw_matches[:500]:
		m = m[0]

		if predict.has_key(classes[m.queryIdx]):
			predict[classes[m.queryIdx]] += 1
		else:
			predict[classes[m.queryIdx]] = 1
	predict = sorted(predict.items(), key = lambda x:x[1])
	for keys in predict[-1:-6:-1]:
		tpl = cv2.imread(tpl_names[keys[0]])
		c = 0
		template = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)
		#template = cv2.GaussianBlur(template,(3,3),0)
		print "------------------------------------"
		print 'SHELF = '+ sh+ ' TEMPLATE = '+ tpl_names[keys[0]]
		(tH, tW) = template.shape[:2]

		scale_space = []
		scales = []
		for scale in np.linspace(0.2, 1.0, 20)[::-1]:
			resized = imutils.resize(gray, width = int(gray.shape[1] * scale))
			if resized.shape[0] < tH or resized.shape[1] < tW:
				break
			r = gray.shape[1] / float(resized.shape[1])
			# edged  = cv2.GaussianBlur(resized,(5,5),0)
			scale_space.append(cv2.matchTemplate(resized, template, cv2.TM_CCOEFF_NORMED))
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
				(startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
				(endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
				g = gray[startY:endY, startX:endX]
				flag = find_obj(template, g)
				if flag:
					cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[c_count], 30)
					output.write(tpl_names[keys[0]]+' '+str(startX)+' '+str(endX)+' '+str(startY)+' '+str(endY)+'\n')
					count = MAX_MIS_PREDICTIONS	
					c+=1
					lower_scale = max(0, ind - 3);
					upper_scale = min(len(scales), ind + 3)
					for k in range(len(scales)):
						startX = max(0, int(maxLoc[0]*scales[ind]/scales[k] - tW/2))
						startY = max(0, int(maxLoc[1]*scales[ind]/scales[k] - tH/2))
						#print '---', k, int(maxLoc[1]*scales[ind]/scales[k]), int(maxLoc[0]*scales[ind]/scales[k]), scale_space[k].shape
						#print startX, endX, startY, endY
						endX = min(int(maxLoc[0]*scales[ind]/scales[k] + tW/2), scale_space[k].shape[1])
						endY = min(int(maxLoc[1]*scales[ind]/scales[k] + tH/2), scale_space[k].shape[0])
						for i in range(startX, endX):
							for j in range(startY,endY):
								scale_space[k][j][i] = 0.0
					
				else:
					count -= 1

			else:
				count -= 1
		c_count = (c_count + 1)%len(COLORS)
	cv2.imwrite("Auto_Results/res" + str(counter) + ".png", image)
	counter += 1
	stop1 = timeit.default_timer()
	print 'Time taken = ', stop1 - start1
stop = timeit.default_timer()
print stop - start 
