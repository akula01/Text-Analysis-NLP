import mmap

from nlp_util import *
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
import os
import random
import pickle


class LabeledParagraphs(object):
    def __init__(self, paragraph_list, labels_list):
        self.labels_list = labels_list
        self.paragraph_list = paragraph_list

    def __iter__(self):
        for idx, paragraph in enumerate(self.paragraph_list):
              yield doc2vec.LabeledSentence(paragraph, [self.labels_list[idx]])


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


def pre_process(paragraphs):
    cleaned_paragraphs = nlp_clean(paragraphs)
    processed_paragraphs = []
    original_paragraphs = []
    for i in range(len(cleaned_paragraphs)):
        if len(cleaned_paragraphs[i]) < 10:
            continue
        processed_paragraphs.append(cleaned_paragraphs[i])
        original_paragraphs.append(paragraphs[i])
    return processed_paragraphs, original_paragraphs


def train_model(paragraphs):
    processed_paragraphs, original_paragraphs = pre_process(paragraphs)

    pickle.dump(original_paragraphs, open('training_data.pkl', 'w'))
    it = LabeledParagraphs(processed_paragraphs, range(0, len(processed_paragraphs)))

    model = doc2vec.Doc2Vec(dm=0, size=100, window=300, min_count=1, workers=4, dbow_words=1, iter=20, alpha=0.025, min_alpha=0.025)

    model.build_vocab(it)
    model.train(it, epochs=model.iter, total_examples=model.corpus_count)

    model.save('doc2vec.model')
    print('model saved')


def train():
    pdf_folder = 'C:/Users/ra407452/Desktop/NSF Eager/IEP Submissions_Ivan Garibay'
    paragraphs = []
    for f in os.listdir(pdf_folder)[0:30]:
        print("Processing file: ", f)
        pdf_file = os.path.join(pdf_folder, f)
        reader = PDFReader(pdf_file)
        paragraphs.extend(reader.get_all_paragraphs())
    train_model(paragraphs)


def test_model(model, paragraphs):
    processed_paragraphs, original_paragraphs = pre_process(paragraphs)
    random_index = random.randint(0, len(processed_paragraphs))
    random_index = 100
    sample_paragraph = processed_paragraphs[random_index]
    print("===========Input===========")
    print(original_paragraphs[random_index])
    print("===========================")
    model.init_sims(replace=False)
    similar_paragraphs = model.docvecs.most_similar(positive=[model.infer_vector(sample_paragraph)], topn=3)
    training_data = pickle.load(open('training_data.pkl', 'r'))
    print("===========Result===========")
    print("Index : ", similar_paragraphs[0][0])
    print(training_data[similar_paragraphs[0][0]])
    print("===========================")


def test():
    #loading the model
    #d2v_model = doc2vec.Doc2Vec.load('doc2vec.model')

    d2v_model = doc2vec.Doc2Vec.load('./trained_models/wiki_sg.tar', mmap=None)
    pdf_folder = 'C:/Users/ra407452/Desktop/NSF Eager/IEP Submissions'
    paragraphs = []
    for f in os.listdir(pdf_folder)[30:31]:
        print("Processing file: ", f)
        pdf_file = os.path.join(pdf_folder, f)
        reader = PDFReader(pdf_file)
        paragraphs.extend(reader.get_all_paragraphs())
    test_model(d2v_model, paragraphs)


if __name__ == '__main__':
    # train()
    test()