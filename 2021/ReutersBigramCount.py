# import modules
import numpy as np
import nltk
#nltk.download('punkt')
import operator
# load list of all news
AllNews = np.load('AllNews.npy')
# initialize empty list of ordered bigrams and ordered dictionaries
OrderedBigrams = []
OrderedDicts = []
# go through list of news
for s in AllNews:
	# tokenize words in each news
	words = nltk.tokenize.word_tokenize(s)
	# create bigrams generator
	bgs = nltk.bigrams(words)
	# get frequency of bigrams
	freq = nltk.FreqDist(bgs)
	# sort words by frequencies
	sortedfreq = sorted(freq.items(), key=operator.itemgetter(1), reverse=True)
	# save ordered dictionary
	OrderedDicts.append(dict(sortedfreq))
	# save ordered bigrams
	ow = []
	for t in sortedfreq:
		ow.append(t[0])
	OrderedBigrams.append(ow)
# save OrderedBigrams and OrderedDicts
np.save('ReutersOrderedBigrams.npy', OrderedBigrams)
np.save('ReutersOrderedBigramDicts.npy', OrderedDicts)
# compile list of all words
AllBigrams = set()
for ow in OrderedBigrams:
	AllBigrams = AllBigrams.union(set(ow))
# create master bigram dictionary
MasterBigramDict = dict()
for key in AllBigrams:
	MasterBigramDict[key] = 0
	for d in OrderedDicts:
		if key in d.keys():
			MasterBigramDict[key] += d[key]
# sort master dictionary
SortedMasterBigramDict = dict(sorted(MasterBigramDict.items(), key=operator.itemgetter(1), reverse=True))
# save AllBigrams, MasterBigramDict, and SortedMasterBigramDict
np.save('ReutersAllBigrams.npy', AllBigrams)
np.save('ReutersMasterBigramDict.npy', MasterBigramDict)
np.save('ReutersSortedMasterBigramDict.npy', SortedMasterBigramDict)
