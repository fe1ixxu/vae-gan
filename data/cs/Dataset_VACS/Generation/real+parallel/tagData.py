import enchant
import unicodedata
import io

'''
 Takes a data file and generates tags/ labels for the data
 For our experiments we used 3 labels:
 - 0 : English
 - 1 : Hindi
 - 2 : Others (Unknown/Number)
 Input: Data file
 Output: Labels File
'''

def is_hindi_char(cc):
	# print(cc)
	# cc=cc.encode("utf-8")
	# print(cc)
	a=unicodedata.category(cc)
	if(a=='Lo'):
		return True
	else:
		return False

def isHindi(seq):
	for i in range(len(seq)):
		if is_hindi_char(seq[i]):
			return True
	return False


def is_number(s):
	try:
		float(s)
		return True
	except ValueError:
		pass

	try:
		unicodedata.numeric(s)
		return True
	except (TypeError, ValueError):
		pass
	return False

d=enchant.Dict("en_US")

fileip=io.open('./data.txt','r')
fileop=open('./labels.txt','w')
# file1=io.open('./vae_data.txt','w')


#print(is_number(word))
#exit()
line=fileip.readline()
# print(line)

while line:
	label=[]
	# print(line[-1],"---------")
	# line=line.replace("."," .")
	wordlist=line.strip().split()

	print(wordlist)
	# file1.write(line)	
	for word in wordlist:
		#print(word)
		#word="3"
		print(word)
		# if(word==''):
		# 	continue
		if(word=="."):
			fileop.write("2"+" ")
			label.append("2")
			print("2")
			continue
		elif(word=='HASHTAG' or word=='URL' or word=='MENTION'):
			fileop.write("2"+" ")
			label.append("2")
			print("2")
			continue
		elif is_number(word):
			#print("yes")
			fileop.write("2"+" ")
			label.append("2")
			print("2")
			continue
		elif(isHindi(word)):
			#print("hindi")
			fileop.write("1"+" ")
			label.append("1")
			print("1")
			continue
		elif d.check(word):
			#print("no")
			fileop.write("0"+" ")
			label.append("0")
			print("0")
			continue
		else:
			label.append("2")
			fileop.write("2"+" ")
			print("2")
			

		# fileop.write("2"+" ")

	#exit(0)
	print(len(wordlist),len(label))
	# exit()
	line=fileip.readline()
	fileop.write("\n")
fileip.close()
fileop.close()
# file1.close()
