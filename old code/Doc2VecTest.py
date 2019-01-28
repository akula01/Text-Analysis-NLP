# Import libraries

import gensim
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from os import listdir
from os.path import isfile, join


def nlp_clean(data):
    tokenizer = RegexpTokenizer(r'\w+')
    stopword_set = set(stopwords.words('english'))
    new_data = []
    for d in data:
       new_str = d.lower()
       dlist = tokenizer.tokenize(new_str)
       dlist = list(set(dlist).difference(stopword_set))
       new_data.append(dlist)
    return new_data


#This function does all cleaning of data using two objects above


# Load data

data = ["This is a sentence", "This is another sentence"]

# Transform data (you can add more data preprocessing steps)

docs = []
analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
for i, text in enumerate(doc1):
    words = text.lower().split()
    tags = [i]
    docs.append(analyzedDocument(words, tags))

# Train model (set min_count = 1, if you want the model to work with the provided example data set)

model = doc2vec.Doc2Vec(size=100, window=300, min_count=1, workers=4, dbow_words=1)

model.build_vocab(docs)

#training of model
for epoch in range(100):
    print 'iteration ' +str(epoch+1)
    model.train(docs)
    model.alpha -= 0.002
    model.min_alpha = model.alpha

#saving the created model
model.save('doc2vec.model')
print 'model saved'

# Get the vectors

print(len(model.docvecs[0]))
print(model.docvecs[0])
print(model.docvecs[1])