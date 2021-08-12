#ETEtoolkit
from fastopic import Seqpic 
import math 

class Statpic:
	def __init__(self, seqpic):
		#self.prob = self.getProb(seqpic.makeHist(), len(seqpic.picture)) 
		self.prob = self.getProb(seqpic.makeHist(), seqpic.picture.size) 
		self.average = self.getAverage(seqpic.makeHist()) 
		self.variance = self.getVariance(seqpic.makeHist())
		self.desv = math.sqrt(self.variance)
		self.skewness = self.getSkewness(seqpic.makeHist())
		self.kurtosis = self.getKurtosis(seqpic.makeHist())
		self.energy = self.getEnergy()
		self.entropy = self.getEntropy()
	
	def getProb(self, histo,size):
		prob = []
		for i in range(len(histo)):
			prob.append(histo[i][1]/size)	
		return prob

	def getAverage(self, histo):
		average = 0
		for i in range(len(self.prob)):
			average += histo[i][0]*self.prob[i]
		return average

	def getVariance(self,histo):
		variance = 0
		for i in range(len(self.prob)):
			variance += pow(histo[i][0]-self.average,2) * self.prob[i]
		return variance

	def getSkewness(self,histo):
		skewness = 0
		for i in range(len(self.prob)):
			skewness += pow(histo[i][0]-self.average,3) * self.prob[i]
		skewness = skewness / pow(self.desv,3)
		return skewness

	def getKurtosis(self, histo):
		kurtosis = 0
		for i in range(len(self.prob)):
			kurtosis += pow(histo[i][0]-self.average,4) * self.prob[i] 
		kurtosis =(kurtosis / pow(self.desv,4))-3
		return kurtosis

	def getEnergy(self):
		energy = 0
		for i in range(len(self.prob)):
			energy += pow(self.prob[i],2)
		return energy
		
	def getEntropy(self):
		entropy = 0
		for i in range(len(self.prob)):
			entropy += self.prob[i] * math.log(self.prob[i],2)
		return entropy*-1
		
	def makeVecFeatures(self):
		vec = []
		vec.append(self.skewness)
		vec.append(self.kurtosis)
		vec.append(self.energy)
		vec.append(self.entropy)
		return vec
