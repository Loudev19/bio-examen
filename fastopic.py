from Bio import SeqIO
import numpy as np
from matplotlib import pyplot as plt
import os

class Seqpic:

	def __init__(self, path, outpath_):
		self.title = ""
		self.alpha = {'AA':[1,0],'AG':[17,0],'AC':[34,0],'AT':[51,0],'GA':[68,0],'GG':[85,0],'GC':[102,0],'GT':[119,0],'CA':[136,0],'CG':[153,0],'CC':[170,0],'CT':[187,0],'TA':[204,0],'TG':[221,0],'TC':[238,0],'TT':[255,0]}
		self.picture = self.makePic(path)
		self.outpath = outpath_

	def makePic(self, path):
		data = self.datacleaning(path)
		matrix = [[] for _ in range(len(data)//70)]
		it = 0
		for row in range(len(data)//70):
			for _ in range(70):
				if it+1 == len(data):
					break
				if it != (70*(row+1))-1:
					pair = data[it]+data[it+1]
					matrix[row].append(self.alpha[pair][0])
					self.alpha[pair][1] += 1

				it += 1
		return np.array(matrix)

	def datacleaning(self,path, dtype='fasta'):
		sequences = SeqIO.parse(path,dtype)
		for record in sequences:
			data = str(record.seq.upper())
			self.title = str(record.description)
		return data

	def picShow(self):
		plt.imshow(self.picture)
		plt.savefig(self.outpath+self.title+'.png')
		plt.show()
	
	def makeHist(self):
		hist = list(self.alpha.values())		
		return hist
	
	def histShow(self):
		first,second = zip(*list(self.alpha.values()))
		plt.bar(first, second, 9)
		plt.xticks(first, first)
		plt.title(self.title)
		plt.ylim(0)
		plt.savefig(self.outpath+self.title+'histo.png')
		plt.show()
		

#mypic = Seqpic("sequence.fasta")

