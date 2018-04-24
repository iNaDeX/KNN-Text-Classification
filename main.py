from __future__ import print_function
import sys
import re
import numpy as np
from numpy import dot
from numpy.linalg import norm
from pyspark import SparkContext

if __name__ == "__main__":
	if len(sys.argv) != 2:
		print("Usage: job <inputFile>", file=sys.stderr)
		exit(-1)
		
	sc = SparkContext(appName="TextClassification")

	# First load up all of the 19997 documents in the corpus
	corpus = sc.textFile(sys.argv[1])

	# Each entry in validLines will be a line from the text file
	validLines = corpus.filter(lambda x : 'id' in x) ## todo: improve, doesn't remove any invalid lines atm (and maybe 'id' is in text content)

	# Now, we transform it into a set of (docID, text) pairs
	keyAndText = validLines.map( lambda x : (x[x.index('id="') + 4 : x.index('" url=')], x[x.index('">') + 2:])) ## doesn't remove final </doc>

	# Now, we split the text in each (docID, text) pair into a list of words
	# After this step, we have a data set with 
	# (docID, ["word1", "word2", "word3", ...])
	# We use a regular expression here to make 
	# sure that the program does not break down on some of the documents

	regex = re.compile('[^a-zA-Z]')

	keyAndListOfWords = keyAndText.map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split())) ## remove all non letter characters

	# Now get the top 20,000 words... 
	# first change (docID, ["word1", "word2", "word3", ...])
	# to ("word1", 1) ("word2", 1)...
	allWords = keyAndListOfWords.flatMap(lambda x: ( (j, 1) for j in x[1] ))

	# Now, count all of the words, giving us ("word1", 1433), ("word2", 3423423), etc.
	allCounts = allWords.reduceByKey(lambda a,b : a+b)

	# Get the top 20,000 words in a local array in a sorted format based on frequency 
	topWords = allCounts.top(20000, key=lambda x: x[1])

	# We'll create a RDD that has a set of (word, dictNum) pairs
	# start by creating an RDD that has the number 0 through 20000
	# 20000 is the number of words that will be in our dictionary
	twentyK = sc.parallelize(range(20000))

	# Now, we transform (0), (1), (2), ... to ("mostcommonword", 1) 
	# ("nextmostcommon", 2), ...
	# the number will be the spot in the dictionary used to tell us 
	# where the word is located
	dictionary = twentyK.map (lambda x : (topWords[x][0], x))

	# Next step, we get a RDD that has, for each 
	# (docID, ["word1", "word2", "word3", ...]),
	# ("word1", docID), ("word2", docId), ...
	allWords = keyAndListOfWords.flatMap(lambda x: ((j, x[0]) for j in x[1]))

	# Now join and link them, to get a set of ("word1", (dictionaryPos, docID)) pairs
	allDictionaryWords = dictionary.join(allWords)

	# Now, we drop the actual word itself to get a set of (docID, dictionaryPos) pairs
	justDocAndPos = allDictionaryWords.map (lambda x: (x[1][1], x[1][0]))

	# Now get a set of (docID, [dictionaryPos1, dictionaryPos2, dictionaryPos3...]) pairs
	allDictionaryWordsInEachDoc = justDocAndPos.groupByKey()

	################### TASK 2

	# Now, extract the newsgroupID, so that on input we have a set of
	# (docID, [dictionaryPos1, dictionaryPos2, dictionaryPos3...]) pairs, 
	# but on output we 
	# have a set of ((docID, newsgroupID) [dictionaryPos1, dictionaryPos2, dictionaryPos3...]) pairs
	# The newsgroupID is the name of the newsgroup extracted from 
	# the docID... for example 
	# if the docID is "20_newsgroups/comp.graphics/37261" then 
	# the newsgroupID will be "/comp.graphics/"

	regex = re.compile('/.*?/')

	allDictionaryWordsInEachDocWithNewsgroup = allDictionaryWordsInEachDoc.map (lambda x: ((x[0], regex.search(x[0]).group (0)), x[1]))

	# The following function gets a list of dictionaryPos values, 
	# and then creates a TF vector
	# corresponding to those values... for example, 
	# if we get [3, 4, 1, 1, 2] we would in the
	# end have [0, 2/5, 1/5, 1/5, 1/5] because 0 appears zero times, 
	# 1 appears twice, 2 appears once, etc.

	def buildArray (listOfIndices):
		returnVal = np.zeros (20000)
		for index in listOfIndices:
			returnVal[index] = returnVal[index] + 1
		mysum = np.sum (returnVal)
		returnVal = np.divide (returnVal, mysum)
		return returnVal

	# The following line this gets us a set of 
	# ((docID, newsgroupID) [dictionaryPos1, dictionaryPos2, dictionaryPos3...]) pairs
	# and converts the dictionary positions to a bag-of-words numpy array... 

	allDocsAsNumpyArrays = allDictionaryWordsInEachDocWithNewsgroup.map (lambda x: (x[0], buildArray (x[1])))

	# Now, create a version of allDocsAsNumpyArrays where, in the array, 
	# every entry is either zero or one.  
	# A zero means that the word does not occur,
	# and a one means that it does.
	def zeroOneIt(x):
		x[x > 0] = 1
		return x
		
	zeroOrOne = allDocsAsNumpyArrays.map(lambda x: (x[0], zeroOneIt(x[1]))) # make all elements > 0 be 1

	# Now, add up all of those arrays into a single array, where the 
	# i^th entry tells us how many
	# individual documents the i^th word in the dictionary appeared in
	dfArray = zeroOrOne.reduce (lambda x1, x2:(("", np.add (x1[1], x2[1]))))[1]

	# Create an array of 20,000 entries, each entry with the value 19997.0 (number of docs)
	multiplier = np.full (20000, 19997.0)

	# Get the version of dfArray where the i^th entry is the inverse-document frequency for the
	# i^th word in the corpus
	idfArray = np.log (np.divide (multiplier,dfArray))

	# Finally, convert all of the tf vectors in allDocsAsNumpyArrays to tf * idf vectors
	allDocsAsNumpyArrays = allDocsAsNumpyArrays.map (lambda x: (x[0], np.multiply (x[1], idfArray)))

	############## TASK 3
	# replacement for scipy cosine distance function, not available on EMR
	def cos_sim(a,b):
		return dot(a, b)/(norm(a)*norm(b))

	# Finally, we have a function that returns the prediction for the label of a string, using a kNN algorithm
	def getPrediction (textInput, k):
		# Create an RDD out of the textIput
		myDoc = sc.parallelize (('', textInput))
		# Flat map the text to (word, 1) pair for each word in the doc
		wordsInThatDoc = myDoc.flatMap (lambda x : ((j, 1) for j in regex.sub(' ', x).lower().split()))
		#
		# This will give us a set of (word, (dictionaryPos, 1)) pairs
		allDictionaryWordsInThatDoc = dictionary.join (wordsInThatDoc).map (lambda x: (x[1][1], x[1][0])).groupByKey ()
	   
		# Get tf array for the input string
		myArray = buildArray (allDictionaryWordsInThatDoc.top (1)[0][1])
		#
		# Get the tf * idf array for the input string
		myArray = np.multiply (myArray, idfArray)
		#
		# Get the distance from the input text string to all database documents, using cosine similarity
		#distances = allDocsAsNumpyArrays.map (lambda x: (x[0], 1 - distance.cosine(myArray,x[1]))) ## needs scipy, which is not on EMR
		distances = allDocsAsNumpyArrays.map (lambda x: (x[0], cos_sim(myArray,x[1]))) # compute similarity and not distance ! (that's why not 1 - x)
		#
		# get the top k distances
		topK = distances.top (k, lambda x : x[1]) ## we compute similarity and not distances
		#
		# and transform the top k distances into a set of (newsgroupID, 1) pairs
		newsgroupsRepresented = sc.parallelize (topK).map (lambda x : (x[0][1], 1)) ## only keep groupID for the reduceByKey to work properly
		#
		# now, for each newsgroupID, get the count of the number of times this newsgroup appeared in the top k
		numTimes = newsgroupsRepresented.reduceByKey(lambda a,b: a+b)
		# Return the top 1 of them.  
		return numTimes.top (1, lambda x: x[1])

	print(getPrediction('God Jesus Allah', 30)) # 	[('/soc.religion.christian/', 11)]
	print(getPrediction("How many goals Vancouver score last year?", 30)) # [('/rec.sport.hockey/', 23)]

	sc.stop()
