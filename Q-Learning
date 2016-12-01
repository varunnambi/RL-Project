# Varun Nambikrishnan 2016
# November 11th, 2016
#
# Program that implements Q-learning to find an optimal policy for three Markov Decision Processes.
# The first is a 10x10 (100 states) grid world with 4 actions, the second is the MountainCarContinous-v0 environment
# from OpenAI Gym with altered parameters with 50,000 states and 7 actions, and the third is an unknown MDP with
# 1,757,600 states and 8 actions.
# 
#


import csv
import collections
import math

# Algorithm uses global function approximation to generalize for state-action pairs without data.
# Additionally, stochiastic gradient descent is used to update the weight vector, and then the policy
# is outputted to a file using the updated weight vector after looping through the data in the csv

def Qlearning(filename, numActions, disFactor, maxStates):
	weightVec = collections.defaultdict(float)
	extract = getExtractor(filename)
	with open(filename, 'rb') as csvfile:
		filereader = csv.reader(csvfile, dialect='excel')
		filereader.next()
		for row in filereader:
			curfeats = extract(int(row[0]), int(row[1]))
			Qopt = dotproduct(weightVec, curfeats)
			Vopt = findQopt(int(row[3]), numActions, weightVec, 0, extract)
			predMinusTarg = Qopt - (int(row[2]) + disFactor * Vopt)
			scalarMult(.001 * predMinusTarg, curfeats) #constant is learning rate
			subtractSparseDicts(weightVec, curfeats)
	pol = findPolicy(weightVec, numActions, maxStates, extract) 
	with open('small.policy', 'w') as f: # Change to small, medium, or large 
		for i in range(1, maxStates + 1):
			f.write("%s\n" % str(pol[i-1]))

def getExtractor(filename):
	if(filename == "Downloads/small.csv"):
		return getSmallExtractor
	if(filename == "Downloads/medium.csv"):
		return getMediumExtractor
	if(filename == "Downloads/large.csv"):
		return getLargeExtractor
	return

		
def getSmallExtractor(s,a):
	features = collections.defaultdict(float)
	nextLoc = calcNextLoc(s,a)
	features['distanceFrom78'] = 1.0/distance(nextLoc, 78)
	return features

def getMediumExtractor(s,a):
	features = collections.defaultdict(float)
	pos = (s - 1) % 500
	v = (s - 1)/500
	features[(pos, v, a)] = 1
	features[(v, a)] = 1
	return features

def getLargeExtractor(s, a):
	features = collections.defaultdict(float)
	features[(s, a)] = 1
	return features


def calcNextLoc(s,a):
	if(a == 1):
		return s - 1
	if(a == 2):
		return s + 1
	if(a == 3):
		return s + 10
	return s - 10

def xycord(s):
	return [s % 10, s/10]

def distance(s, s2):
	if(s == s2):
		return .5
	scord = xycord(s)
	s2cord = xycord(s2)
	return math.sqrt(abs((s2cord[0] - scord[0])**2 + (s2cord[1] - scord[1])**2))

def closerToGoal(s, next):
	scord = xycord(s)

def dotproduct(d1,d2):
	return sum(d1[key] * d2[key] for key in d2)

def subtractSparseDicts(d1,d2):
	for key in d2:
		d1[key] -= d2[key]

def scalarMult(n, d):
	for key in d:
		d[key] = d[key] * n

# Finds Q value of the action that maximizes the reward
def findQopt(s, numActions, wv, returnAction, extract):
	Qopt = 0
	maxA = 0
	for i in range(1,numActions + 1):
		curfeat = extract(s, i)
		Qi = dotproduct(wv, curfeat)
		if(i == 1 or Qi > Qopt):
			Qopt = Qi
			maxA = i
	if(returnAction == 0):
		return Qopt
	return maxA

# Method to find optimal policy, after weight vector is fully updated
def findPolicy(wv, numActions, maxStates, extract):
	policy = [None] * (maxStates)
	for i in range(1, maxStates+1):
		maxA = findQopt(i,numActions, wv, 1, extract)
		policy[i-1] = maxA
	return policy


## Uncomment whichever project you wish to run and change
## output file (in Qlearning) to small, medium, or large
Qlearning("Downloads/small.csv", 4, .95, 100)
#Qlearning("Downloads/medium.csv", 7, 1, 50000)
#Qlearning("Downloads/large.csv", 8, .99, 1757600)
