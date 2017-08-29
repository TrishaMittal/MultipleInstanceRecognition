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
	#print pred				
	for p in pred:
		p = p.split()
		if(predict.has_key(int(p[0]))):
			flag = True
			for rect in predict[int(p[0])]:
				overlap = findOverlap(rect, [float(p[1]),float(p[2]),float(p[3]),float(p[4])])
				#print rect, [float(p[1]),float(p[2]),float(p[3]),float(p[4])]
				if overlap > 0.8:
					flag = False
					correct_pos += 1
					break
			if(flag):
				false_pos += 1
		else:
			false_pos += 1	
	return correct_pos, false_pos
		
gt = open('annotations/a_tea.annotation', 'r')
shelves = {}
for line in gt.readlines():
	line = line.split()
	if len(line)>0:
		if line[0] == 'file:':
			temp = line[1].split('/')
			temp = temp[-1].split('.')
			shelves[temp[0]] = []
			curr = temp[0]
		else:
			if(line[0]=='bbox:'):
				temp = line[1].split(',')
				xmax = float(temp[0])+float(temp[2])
				ymax = float(temp[1])+float(temp[3])
				template = ' '+temp[0]+' '+str(xmax)+' '+temp[1]+' '+str(ymax)+'\n'

			elif(line[0] == 'item:'):
				template = line[1] + template
				#print template
				shelves[curr].append(template)
				#print shelves[curr]

out = open('output_full_tea_nms')
predicted = []
flag = False
truth = []
avg = 0
count = 0
for line in out.readlines():
	if(line[0] == 'S'):
		line = line.split()
		if(len(predicted)!=0):
			correct_pos, false_pos = compare(predicted, truth, correct_pos, false_pos)
			avg += correct_pos/(1.0*false_pos + 1.0*correct_pos)
			count += 1
		tmp = line[1].split('/')
		tmp = tmp[-1].split('.')
		if shelves.has_key(tmp[0]):
			flag = True
			truth = shelves[tmp[0]]
		else:
			flag = False
		predicted = []
	else:
		if(flag):
			#print line
			predicted.append(line)
		else:
			false_pos += 1
		
if(len(predicted)!=0):
	correct_pos, false_pos = compare(predicted, truth, correct_pos, false_pos)
print correct_pos, false_pos
print 'precision = ',correct_pos/(1.0*false_pos + 1.0*correct_pos)
print avg/count
