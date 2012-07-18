bangla_pos_tagger
=================

POS Tagger for Bangla language based on Conditional Random Fields

Usage
=====
1. Install the module

	python setup.py install
  
2. Code
	
	import bangla_pos_tagger
	btagger=bangla_pos_tagger.BanglaTagger()
	
	#Query is an array of Bangla words
	btagger.pos_tag(query)
	
	#term is a single Bengali Term
	btagger.get_tag(term)
	
  
where query is a tokenized words for a given Bangla Sentence.

Observations
============
* Unigram Based Tagger gives approximately 60-65% accuracy.
* Adding Bigram, and Trigram based taggers following the same increases the accuracy to some extent.
* Adding an affix based tagger, improves the accuracy a bit.

Note: In the "accuracy.txt" file in the analyzed_data directory. Only the relevant results have been added which were giving really good accuracies. The analysis is similar to that of the blog.

Data Sets
============
Dataset "bangla.pos" as comes packaged with NLTK
Dataset "nltr" from nltr.org
Dataset "msr" containing files [1-9,e1,e2].xml as described here - http://www.ldc.upenn.edu/Catalog/catalogEntry.jsp?catalogId=LDC2010T16
Relevant Blog Posts
-------------------
1. http://streamhacker.com/2008/12/29/how-to-train-a-nltk-chunker/
2. http://streamhacker.com/2008/11/10/part-of-speech-tagging-with-nltk-part-2/
3. http://streamhacker.com/2008/12/03/part-of-speech-tagging-with-nltk-part-3/
4. http://streamhacker.com/2010/04/12/pos-tag-nltk-brill-classifier/