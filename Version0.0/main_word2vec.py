from nlp_util import *
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
import os
import random
import gensim
import pickle


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


def evaluate(paragraphs):
    processed_paragraphs, original_paragraphs = pre_process(paragraphs)

    pickle.dump(original_paragraphs, open('original_data.pkl', 'w'))
    pickle.dump(processed_paragraphs, open('processed_data.pkl', 'w'))

    model = gensim.models.KeyedVectors.load_word2vec_format('C:/Users/ra407452/Desktop/NSF Eager/TextAnalysis/trained_models/GoogleNews-vectors-negative300.bin.gz', binary=True)

    random_index = random.randint(0, len(processed_paragraphs))

    input_paragraph = processed_paragraphs[random_index]

    results = {}
    for i in range(len(processed_paragraphs)):
        if i != random_index:
            similairty = calculate_similarity(processed_paragraphs[i], input_paragraph, model)
            results[i] = similairty
    sorted_results = sorted(results, key=results.get, reverse=True)
    print('****** Input ******')
    print(original_paragraphs[random_index])
    print('****** Output ******')
    print(original_paragraphs[sorted_results[0]])


def calculate_similarity(paragraph_1, paragraph_2, model):

    similarity = 0.0
    for i in paragraph_1:
        for j in paragraph_2:
            if i in model.vocab and j in model.vocab:
                similarity += model.similarity(i, j)
    return similarity


def run():
    pdf_folder = 'C:/Users/ra407452/Desktop/NSF Eager/IEP Submissions'
    paragraphs = []
    for f in os.listdir(pdf_folder):
        print("Processing file: ", f)
        pdf_file = os.path.join(pdf_folder, f)
        reader = PDFReader(pdf_file)
        paragraphs.extend(reader.get_all_paragraphs())
    evaluate(paragraphs)



if __name__ == '__main__':
    run()