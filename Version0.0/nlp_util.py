from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import hashlib
from input_util import *
from gensim.models import Word2Vec
from nltk.corpus import wordnet as wn


class TextSummarizer:

    def __init__(self, pdf_file):
        self.pdf_file = pdf_file

    def build_word2vec_model(self):
        pdf_reader = PDFReader(self.pdf_file)
        stopWords = set(stopwords.words("english"))
        sentences = []
        for paragraph in pdf_reader.get_all_paragraphs():
            sentences.extend(sent_tokenize(paragraph))
        self.word2vec_model = Word2Vec(sentences)

    def build_freq_table(self):
        pdf_reader = PDFReader(self.pdf_file)
        stopWords = set(stopwords.words("english"))
        freqTable = dict()
        for paragraph in pdf_reader.get_all_paragraphs():
            if len(paragraph) == 0:
                continue
            words = word_tokenize(paragraph)
            for word in words:
                word = word.lower()
                if len(word) <= 2:
                    continue
                if word in stopWords:
                    continue
                if word in freqTable:
                    freqTable[word] += 1
                else:
                    freqTable[word] = 1
        return freqTable

    def get_similarity(self, key_word, freq_table, paragraph):

        words = word_tokenize(paragraph)
        sum_freq = sum(freq_table.values())
        overall_similarity = 0.0
        for word in words:
            word = word.lower()
            if word in freq_table.keys():
                freq = freq_table[word]
                word_synsets = wn.synsets(word)
                key_word_synsets = wn.synsets(key_word)
                if len(word_synsets) == 0:
                    continue
                similarity = key_word_synsets[0].path_similarity(word_synsets[0])
                if similarity is None:
                    continue
                similarity = similarity * float(freq)/sum_freq
                overall_similarity += similarity
        return overall_similarity

    def summarize_paragraph(self, freqTable, paragraph):
        sentences = sent_tokenize(paragraph)
        sentenceValue = dict()
        for sentence in sentences:
            hash_object = hashlib.sha1(sentence.encode('utf-8'))
            hex_dig = hash_object.hexdigest()
            for wordValue in freqTable:
                if wordValue in sentence.lower():
                    if hex_dig in sentenceValue.keys():
                        sentenceValue[hex_dig] += freqTable[wordValue]
                    else:
                        sentenceValue[hex_dig] = freqTable[wordValue]

        sumValues = 0
        for sentence in sentenceValue.keys():
            sumValues += sentenceValue[sentence]

        if len(sentenceValue.keys()) == 0:
            return ''

        # Average value of a sentence from original text
        average = int(sumValues / len(sentenceValue))

        summary = ''
        for sentence in sentences:
            hash_object = hashlib.sha1(sentence.encode('utf-8'))
            hex_dig = hash_object.hexdigest()
            if hex_dig in sentenceValue.keys() and sentenceValue[hex_dig] > (1.5 * average):
                summary += " " + sentence

        return summary