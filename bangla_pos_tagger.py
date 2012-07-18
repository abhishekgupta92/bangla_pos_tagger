import nltk
from nltk.tag import brill
import params
import os
from xml.dom.minidom import parse

class BanglaTagger:

	def __init__(self,msr_data=True,msr_data_location=params.msr_dataset):

		tagged_sents=nltk.corpus.indian.tagged_sents(fileids="bangla.pos")
		filtered_sents1=[]
		for i,sent in enumerate(tagged_sents):
			try:
				nltk.tag.UnigramTagger([sent])
				filtered_sents1.append(sent)
			except ValueError:
				pass

		filtered_sents2=[]

		if msr_data==True:
			cwdir=os.getcwd()
			listing=os.listdir(msr_data_location)
			os.chdir(msr_data_location)
			for fileName in listing:
				xmlTree=parse(fileName)
				nodes=xmlTree.getElementsByTagName('sentence')
				try:
					for node in nodes:
						pairs=node.childNodes[0].nodeValue.split()
						pairs=map(lambda a:a.split("\\"),pairs)
						pairs=map(lambda (a,b):(a.encode("utf-8"),b.encode().split(".")[0]),pairs)
						nltk.tag.UnigramTagger([pairs])
						filtered_sents2.append(pairs)
				except ValueError:
					pass
			os.chdir(cwdir)
			
		total_set=filtered_sents1+filtered_sents2

		aubt_tagger=[nltk.tag.AffixTagger,nltk.tag.UnigramTagger,nltk.tag.BigramTagger,nltk.tag.TrigramTagger]
		brill_templates = [
			brill.SymmetricProximateTokensTemplate(brill.ProximateTagsRule, (1,1)),
			brill.SymmetricProximateTokensTemplate(brill.ProximateTagsRule, (2,2)),
			brill.SymmetricProximateTokensTemplate(brill.ProximateTagsRule, (1,2)),
			brill.SymmetricProximateTokensTemplate(brill.ProximateTagsRule, (1,3)),
			brill.SymmetricProximateTokensTemplate(brill.ProximateWordsRule, (1,1)),
			brill.SymmetricProximateTokensTemplate(brill.ProximateWordsRule, (2,2)),
			brill.SymmetricProximateTokensTemplate(brill.ProximateWordsRule, (1,2)),
			brill.SymmetricProximateTokensTemplate(brill.ProximateWordsRule, (1,3)),
			brill.ProximateTokensTemplate(brill.ProximateTagsRule, (-1, -1), (1,1)),
			brill.ProximateTokensTemplate(brill.ProximateWordsRule, (-1, -1), (1,1))
		]

		btrainer = nltk.tag.brill.FastBrillTaggerTrainer(self.backoff_tagger(total_set,aubt_tagger), brill_templates)
		self.tagger = btrainer.train(total_set, max_rules=300, min_score=3)

	''' Supporting Functions '''
	def backoff_tagger(self, tagged_sents, tagger_classes, backoff=None):
		if not backoff:
			backoff = tagger_classes[0](tagged_sents)


		for cls in tagger_classes:
			tagger = cls(tagged_sents, backoff=backoff)
			backoff = tagger

		return backoff

	def get_tag(self,term):
		tag=self.tagger.tag([term])[0][1]
		if tag==None:
			tag=self.tagger.tag([term.encode("utf-8")])[0][1]
		return tag

	def pos_tag(self, query):
		return self.tagger.tag(query)
		
if __name__ == "__main__":
	btagger=BanglaTagger()
	while True:
		try:
			query=raw_input("Please enter the word to be tagged.\nPress Ctrl+c to break.\n")
			print btagger.get_tag(query)
		except KeyboardInterrupt:
			break
