false_pos = 0
correct_pos = 0
def findOverlap(r1, r2):
	x_overlap = max(0, min(r1[1],r2[1]) - max(r1[0],r2[0]));
	y_overlap = max(0, min(r1[3],r2[3]) - max(r1[2],r2[2]));
	return x_overlap * y_overlap;

def compare(pred, truth, correct_pos, false_pos):
	predict = {}
	for t in truth:
		t = t.split()
		num = int(t[0])
		if(num>-1):
			if(predict.has_key(num)):
				predict[num].append([float(t[1]),float(t[2]),float(t[3]),float(t[4])])
			else:
				predict[num] = [[float(t[1]),float(t[2]),float(t[3]),float(t[4])]]				
	for p in pred:
		p = p.split()
		if(predict.has_key(int(p[0]))):
			for rect in predict[int(p[0])]:
				overlap = findOverlap(rect, [float(p[1]),float(p[2]),float(p[3]),float(p[4])])
				if overlap > 0:
					#print overlap
					correct_pos += 1
				else:
					false_pos += 1
		else:
			false_pos += 1	
	return correct_pos, false_pos
		
gt = open('acc_tea', 'r')
shelves = {}
for line in gt.readlines():
	if line[0] == 'S':
		line = line.split()
		shelves[line[1]] = []
		curr = line[1]
	else:
		shelves[curr].append(line)
		print shelves[curr]

out = open('output5','r')
predicted = []
flag = False
truth = []
for line in out.readlines():
	if(line[0] == 'S'):
		line = line.split()
		if(len(predicted)!=0):
			correct_pos, false_pos = compare(predicted, truth, correct_pos, false_pos)
		if shelves.has_key(line[1]):
			flag = True
			truth = shelves[line[1]]
		else:
			flag = False
		predicted = []
	else:
		if(flag):
			predicted.append(line)
		else:
			false_pos += 1
		
if(len(predicted)!=0):
	correct_pos, false_pos = compare(predicted, truth, correct_pos, false_pos)
print correct_pos, false_pos
print 'precision = ',correct_pos/(1.0*false_pos + 1.0*correct_pos)
