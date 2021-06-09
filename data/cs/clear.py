files = ["used_for_gen"] #["train.pos", "dev.pos", "test.pos","train.neg", "dev.neg", "test.neg"]
def remove_repeat_words(line, sym):
	line = line.split(" ")
	for i, word in enumerate(line):
		line[i] = sym if sym in word else word
	seen = False
	res = []
	for i, word in enumerate(line):
		if word != sym:
			seen = False
			res.append(word.lower())
		elif word == sym and not seen:
			res.append(word)
			seen = True

	res = " ".join(res)
	return res

for file in files:
	with open(file, encoding="utf-8") as fr:
		with open("../cs-norepeat/"+file, "w", encoding="utf-8") as fw:
			line = fr.readline()
			while(line):
				line = remove_repeat_words(line, "HASHTAG")
				line = remove_repeat_words(line, "MENTION")
				line = remove_repeat_words(line, "URL")
				line = line.strip()
				fw.writelines([line, "\n"])
				line = fr.readline()
			
