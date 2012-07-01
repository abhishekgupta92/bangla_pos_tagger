import nltk
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


tagged_sents=nltk.corpus.indian.tagged_sents(fileids="bangla.pos")
filtered_sents1=[]
for i,sent in enumerate(tagged_sents):
	try:
		tagger=nltk.tag.UnigramTagger([sent])
		filtered_sents1.append(sent)
	except ValueError:
		pass

total_set=filtered_sents1

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

btrainer = nltk.tag.brill.FastBrillTaggerTrainer(backoff_tagger(total_set,aubt_tagger), brill_templates)
tagger = btrainer.train(total_set, max_rules=300, min_score=3)

def pos_tag(query):
	return tagger.tag(query)
