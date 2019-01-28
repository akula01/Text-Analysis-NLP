import os
import string
import gensim
from gensim import corpora
from nltk import RegexpTokenizer, pprint
from nltk.stem.wordnet import WordNetLemmatizer
from nlp_util import *

def clean(paragraph):
    stop = set(stopwords.words('english'))
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()
    stop_free = " ".join([i for i in paragraph.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    cleaned = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return cleaned

def train():
    pdf_folder = 'C:/Users/ra407452/Desktop/NSF Eager/IEP Submissions'
    paragraphs = []
    for f in os.listdir(pdf_folder):
        print("Processing file: ", f)
        pdf_file = os.path.join(pdf_folder, f)
        reader = PDFReader(pdf_file)
        paragraphs.extend(reader.get_all_paragraphs())
    processed_paragraphs = [clean(paragraph).split() for paragraph in paragraphs]
    dictionary = corpora.Dictionary(processed_paragraphs)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in processed_paragraphs]
    Lda = gensim.models.ldamodel.LdaModel
    ldamodel = Lda(doc_term_matrix, num_topics=10, id2word=dictionary, passes=250)
    pprint(ldamodel.print_topics(num_topics=10, num_words=10))

    # testing
    test_paragraph = clean(paragraphs[10]).split()
    bow = dictionary.doc2bow(test_paragraph)
    pprint(ldamodel[bow])

if __name__ == '__main__':
    train()