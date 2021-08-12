import math

class Pdist:
	def __init__(self, in_matrix,_metric):
		self.matrix = in_matrix
		self.vector = self.getVectorDist(int(_metric)) 
		self.vectorId = self.getVectorDistId(int(_metric)) 
		#self.euclidean = self.euclideanDist()
	
	def euclideanDist(self, vecA, vecB):
		suma = 0
		for i in range(len(vecA)):
			suma += math.pow(vecA[i]-vecB[i],2)
		return math.sqrt(suma)
	
	def minkowski(self, vecA, vecB, r):
		suma = 0
		for i in range(len(vecA)):
			suma += math.pow(abs(vecA[i]-vecB[i]),r)
		if(r == 2):
			return math.sqrt(suma)
		else:
			return suma
	
	def getVectorDist(self,metric):
		n = len(self.matrix)
		vecDist = []
		for i in range(n-1):
			for j in range(i+1,n):
				if metric == 1:	
					vecDist.append(self.minkowski(self.matrix[i], self.matrix[j],1))
				elif metric == 2:
					vecDist.append(self.euclideanDist(self.matrix[i], self.matrix[j]))
		return vecDist
	
	def getVectorDistId(self,metric):
		n = len(self.matrix)
		vecDist = []
		for i in range(n-1):
			for j in range(i+1,n):
				if metric == 1:	
					vecDist.append([i,j,self.minkowski(self.matrix[i], self.matrix[j],1)])
				elif metric == 2:	
					vecDist.append([i,j,self.euclideanDist(self.matrix[i], self.matrix[j])])
		return vecDist

	def getVsAll(self, index):
		vecCompare = []
		for i in range(len(self.vectorId)):
			if self.vectorId[i][0] == index:
				vecCompare.append([self.vectorId[i][1],self.vectorId[i][2]])
			elif self.vectorId[i][1] == index:
				vecCompare.append([self.vectorId[i][0],self.vectorId[i][2]])
		return vecCompare

