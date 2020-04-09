#!/usr/bin/env python

"""
Title:	COMP90024 Cluster and Cloud Computing Assignment 1 - HPC Twitter Processing
Author:	Chuxin Zhou	1061714	zocz@student.unimelb.edu.au
	Dongnan Cao	970205	dongnanc@student.unimelb.edu.au
Task:	To implement a simple, parallelized application leveraging the University of Melbourne HPC facility SPARTAN.
Last editing:	2020-03-09
"""

from mpi4py import MPI
import json
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# This method assigns tasks to each core of each node
# and extracts info (hashtags,languages) from raw json file separately
# input:	raw json file path
# output:	two dicts containing hashtags and languages with their frequency
def process(mpi,path):
	hashtags = {}
	languages = {}
	line_index = 0
	with open(path,'r',encoding='utf-8') as f:
		for line in f:
			line_index += 1
			if mpi.rank == line_index % mpi.size:
				try:
					event  = json.loads(line[:-2])
				except Exception as e1:
					continue
				twitter = Twitter(event)
				if twitter.hashtag:
					hashtag = twitter.hashtag
					for h in hashtag:
						hashtags[h] = hashtags.get(h,0) + 1
				if twitter.lang:
					lang = twitter.lang
					languages[lang] = languages.get(lang,0) + 1
	return hashtags, languages

# This class defines a ranking system
class Ranking:
	def __init__(self,mpi,hashtags,languages):
		hashtags_list = mpi.comm.gather(hashtags,root=0)
		langs_list = mpi.comm.gather(languages,root=0)
		if mpi.rank ==0:
			#print(len(hashtags_list))
			self.combined_hashtags = self.combine(hashtags_list)
			self.combined_langs = self.combine(langs_list)
	def combine(self,uncombined_data):
		combined_data = {}
		for each in uncombined_data:
			for k,v in each.items():
				combined_data[k] = combined_data.get(k,0) + v
		return combined_data
	def get_top_rank(self,n):
		# sort both dicts and return the top n results
		top_hashtags = sorted(self.combined_hashtags.items(),key=lambda items:items[1],reverse=True)
		top_langs = sorted(self.combined_langs.items(),key=lambda items:items[1],reverse=True)
		return top_hashtags[:n],top_langs[:n]	

# This is a Twitter class for extracting hashtags and languages
class Twitter:
	def __init__(self,event):
		if event.get('doc'):
			doc = event['doc']
			if doc.get('entities'):
				entities = doc['entities']
				if entities.get('hashtags'):
					hashtag_list = entities['hashtags']
					self.hashtag = [each['text'].lower() for each in hashtag_list]
				else:
					self.hashtag = None
			if doc.get('lang'):
				self.lang = doc['lang']
			else:
				self.lang = None

# This class is designed for calling mpi method
class Mpi:
    	def __init__(self):
        	self.comm = MPI.COMM_WORLD
        	self.rank = self.comm.Get_rank()
        	self.size = self.comm.Get_size()

if __name__ == "__main__":
	dataset = ["tinyTwitter.json","smallTwitter.json","bigTwitter.json"]
	mpi = Mpi()
	TOP = 10	# top10
	for data in dataset:
		hashtags,languages = process(mpi,data)
		#print(len(hashtags),len(languages),mpi.rank,mpi.size)
		ranking = Ranking(mpi,hashtags,languages)
		if mpi.rank == 0:
			top_hashtags,top_langs = ranking.get_top_rank(TOP)
			print("==================================")
			print('For dataset',data,':')
			print(top_hashtags)
			print(top_langs)
