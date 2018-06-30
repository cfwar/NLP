#alice.txt is Alice In Wonderland 

from nltk.corpus import PlaintextCorpusReader
corpus_root = '/Users/patrickgrayson/Documents'

wordlists = PlaintextCorpusReader(corpus_root, 'alice.txt')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

stopWords = set(stopwords.words("english"))
words = word_tokenize(wordlists.raw('alice.txt'))

freqTable = dict()
for word in words:
    word = word.lower()
    if word in stopWords:
        continue
    if word in freqTable:
        freqTable[word] += 1
    else:
        freqTable[word] = 1

sentences = sent_tokenize(wordlists.raw('alice.txt'))
sentenceValue = dict()

for sentence in sentences:
    for word in freqTable:
        wordScore = freqTable[word]
        if word in sentence.lower():
            if sentence[:10] in sentenceValue:
                sentenceValue[sentence[:10]] += wordScore / len(sentence)
            else:
                sentenceValue[sentence[:10]] = wordScore / len(sentence)

sumValues = 0
for sentence in sentenceValue:
    sumValues += sentenceValue[sentence]

# Average value of a sentence from original text
average = int(sumValues/ len(sentenceValue))

summary = ''
for sentence in sentences:
        if sentence[:10] in sentenceValue and sentenceValue[sentence[:10]] > (1.5 * average):
            summary +=  " " + sentence

print(summary)

from nltk.stem import PorterStemmer
ps = PorterStemmer()
