import nltk
import os,errno
import random

'''
Important Constants
'''
fileNLTRdata="data/nltr"
analyzedDataDir="analyzed_data/"
taggers=[nltk.tag.UnigramTagger, nltk.tag.BigramTagger, nltk.tag.TrigramTagger]


'''
Supporting Functions
'''
def backoff_tagger(tagged_sents, tagger_classes, backoff=None):
    if not backoff:
        backoff = tagger_classes[0](tagged_sents)
        del tagger_classes[0]

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
    fwrite=open(analyzedDataDir+dataset+".stat",'w')
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
    fwrite=open(analyzedDataDir+dataset+".freq",'w')
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

fread=open(fileNLTRdata)
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
    os.makedirs(analyzedDataDir)
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

random.shuffle(total_set)

len_set=len(total_set)
train_length=int(0.8*len_set)

#Prepare a training and a test set
training_set=total_set[:train_length]
test_set=total_set[train_length:]

#RUN DIFFERENT CLASSIFIERS AND REPORT THEIR PERFORMANCES ON THE TEST DATA SET
#CHOOSE A BEST CLASSIFFIER USING BACKOFF TAGGER (Can we do this automatic?)
#HOW TO USE CONDITIONAL RANDOM FIELDS BASED TAGGER TO ACCOMPLISH THIS AND COMPARE RESULTS

tagger = backoff_tagger(training_set, [nltk.tag.UnigramTagger, nltk.tag.BigramTagger, nltk.tag.TrigramTagger])

print accuracy(tagger,test_set)
