import nltk
import os,errno
import random
import params
from nltk.tag import brill

'''
Supporting Functions
'''
def backoff_tagger(tagged_sents, tagger_classes, backoff=None):
	if not backoff:
		backoff = tagger_classes[0](tagged_sents)

	for cls in tagger_classes:
		tagger = cls(tagged_sents, backoff=backoff)
		backoff = tagger

	return backoff

def split_arr(arr):
	try:
		return map(lambda token: (token.split('\\')[0],token.split('\\')[1].split(".")[0]), arr)
	except:
		pass

def stats(arr,dataset):
	numLines=len(arr)
	numWords=len(reduce(lambda x,y:x+y,arr))
	lines=["Stats for dataset "+dataset+"\n","Number of lines: "+str(numLines)+"\n","Number of words: "+str(numWords)+"\n"]
	fwrite=open(params.analyzedDataDir+dataset+".stat",'w')
	fwrite.writelines(lines)
	fwrite.close()

	
def pos_frequency(arr,dataset):
	arr=reduce(lambda x,y:x+y,arr)
	new_arr=map(lambda a:a[1],arr)
	lines=[]
	for e in set(new_arr):
		lines.append((e,new_arr.count(e)))
	lines=sorted(lines,key=lambda a:a[1],reverse=True)
	lines=map(lambda (a,b): str(a)+";"+str(b)+"\n", lines)
	fwrite=open(params.analyzedDataDir+dataset+".freq",'w')
	fwrite.writelines(lines)
	fwrite.close()

def accuracy(tagger,test_set):
	matched=0
	total=0
	
	for data in test_set:
		text=map(lambda a:a[0],data)
		ctags=map(lambda a:a[1],data)
		ntags=map(lambda a:a[1],tagger.tag(text))

		#Compare ctags and ntags for evaluation
		for i in xrange(len(ctags)):
			if ctags[i]==ntags[i]:
				matched=matched+1

		total=total+len(ctags)

	return float(matched)/total

#Filter the sentences from Bangla.pos as provided by nltk (prepared by IIT Kharagpur)
tagged_sents=nltk.corpus.indian.tagged_sents(fileids="bangla.pos")
filtered_sents1=[]

for i,sent in enumerate(tagged_sents):
	try:
		tagger=nltk.tag.UnigramTagger([sent])
		filtered_sents1.append(sent)
	except ValueError:
		pass

fread=open(params.fileNLTRdata)
lines=fread.readlines()
fread.close()

filtered_sents2=[]
lines=map(lambda line:line.split(),lines)
lines=map(lambda line:split_arr(line), lines)

for i,line in enumerate(lines):
	try:
		tagger=nltk.tag.UnigramTagger([line])
		filtered_sents2.append(line)
	except (TypeError, ValueError):
		pass

'''
Generate some statistical data
'''
try:
	os.makedirs(params.analyzedDataDir)
except OSError as exc:
	if exc.errno == errno.EEXIST:
		pass
	else:
		raise

stats(filtered_sents1,"iitk")
stats(filtered_sents2,"nltr")

pos_frequency(filtered_sents1,"iitk")
pos_frequency(filtered_sents2,"nltr")

#Append all the data sources available
total_set=filtered_sents1
#+filtered_sents2 - Don't include this data because of the non standard POS tags that used while tagging

scores=[]
avg_scores=[]
atagger=[nltk.tag.AffixTagger]
utagger=[nltk.tag.UnigramTagger]
btagger=[nltk.tag.BigramTagger]
ttagger=[nltk.tag.TrigramTagger]
ub_tagger=utagger+btagger
ut_tagger=utagger+ttagger
ubt_tagger=ub_tagger+ttagger
aubt_tagger=atagger+ubt_tagger

taggers=[utagger, ub_tagger, ut_tagger, ubt_tagger,atagger,aubt_tagger]
tagger_names=["Unigram Tagger", "Unigram-Bigram Tagger","Unigram Tigram Tagger","Unigram Bigram Trigram Tagger","Affix based tagger","Affix Unigram Bigram Tigram Tagger"]

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

fwrite=open(params.analyzedDataDir+"accuracy.txt",'w')

for j in xrange(len(taggers)):
	avg_scores.append(0)
	scores.append([])
	#tag_classes=copy.copy(taggers[j])
	tag_classes=taggers[j]

	for i in xrange(params.numTrials):
		random.shuffle(total_set)
		len_set=len(total_set)
		train_length=int(0.8*len_set)

		#Prepare a training and a test set
		training_set=total_set[:train_length]
		test_set=total_set[train_length:]
			
		tagger=backoff_tagger(training_set,tag_classes)
				
		scores[j].append(accuracy(tagger,test_set))
		avg_scores[j]=avg_scores[j]+scores[j][i]
		
	avg_scores[j]=float(avg_scores[j])/params.numTrials

	lines=['Tagger:\t'+tagger_names[j]+"\n"]
	
	line=""
	for i in xrange(params.numTrials):
		line=line+str(scores[j][i])+"\t"
	line=line+"\n"
	
	lines.append(line)
	
	lines.append("Accuracy Score:\t"+str(avg_scores[j])+"\n")
	lines.append("\n")
	
	fwrite.writelines(lines)

scores=[]
avg_score=0
for i in xrange(params.numTrials):
	random.shuffle(total_set)
	len_set=len(total_set)
	train_length=int(0.8*len_set)

	#Prepare a training and a test set
	training_set=total_set[:train_length]
	test_set=total_set[train_length:]

	btrainer = nltk.tag.brill.FastBrillTaggerTrainer(backoff_tagger(training_set,aubt_tagger), brill_templates)
	tagger = btrainer.train(training_set, max_rules=300, min_score=3)
	
	scores.append(accuracy(tagger,test_set))
	avg_score=avg_score+scores[i]

avg_score=float(avg_score)/params.numTrials
lines=['Tagger: Brill Based Tagger with AUBT as the trainer Tagger\n']
	
line=""
for i in xrange(params.numTrials):
	line=line+str(scores[i])+"\t"
line=line+"\n"
	
lines.append(line)
	
lines.append("Accuracy Score:\t"+str(avg_score)+"\n")
lines.append("\n")
fwrite.writelines(lines)

fwrite.close()
