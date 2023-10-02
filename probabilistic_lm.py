import spacy
from nltk.util import ngrams
from nltk.lm.preprocessing import pad_both_ends

nlp = spacy.load('en_core_web_sm')
text = "The boy likes books. The books are bank books. Some books are on the table. Some boys books the table. The book has many likes. Many banks like books."

class ProbabilisticLanguageModel():
    def __init__(self, data, max_ngram=3, smoothing=0):
        self.data = data
        self.max_ngram = max_ngram
        self.smoothing = smoothing
        self.count, self.probability = self.preprocess_data(data)

    def preprocess_data(self, data):
        ngram_list = {}
        for i in range(self.max_ngram):
            n = i + 1
            ngram_list[str(n)] = []

        doc = nlp(data)
        for sentence in doc.sents:
            sentence_token = [token.text.lower() for token in sentence if token.is_alpha]
            for i in range(self.max_ngram):
                n = i + 1
                ngram_list[str(n)].extend(list(ngrams(pad_both_ends(sentence_token, n=n), n=n)))

        count = {}
        sum = {}
        for i in range(self.max_ngram):
            n = i + 1
            for element in ngram_list[str(n)]:
                if element[:-1] not in count:
                    count[element[:-1]] = {}
                    sum[element[:-1]] = 0
                if element[-1] not in count[element[:-1]]:
                    count[element[:-1]][element[-1]] = 0
                count[element[:-1]][element[-1]] += 1
                sum[element[:-1]] += 1
        
        probability = {}
        for pre_words in count:
            probability[pre_words] = {}
            total_len = len(count[pre_words])
            for predict_word in count[pre_words]:
                probability[pre_words][predict_word] = (count[pre_words][predict_word] + self.smoothing) / (sum[pre_words] + self.smoothing * total_len)

        return count, probability

    def count_gram(self, predict_word, pre_words):
        predict_word, pre_words = self.preprocess_input(predict_word, pre_words)
        if pre_words not in self.count:
            return 0
        if predict_word not in self.count[pre_words]:
            return 0
        return self.count[pre_words][predict_word]

    def score(self, predict_word, pre_words):
        predict_word, pre_words = self.preprocess_input(predict_word, pre_words)
        if pre_words not in self.count:
            return 0
        if predict_word not in self.count[pre_words]:
            return 0
        return self.probability[pre_words][predict_word]
    
    def score_sentence(self, sentence, interpolation=True, backoff=False, weight=[1, 1, 1]):
        if interpolation:
            assert len(weight) == self.max_ngram
            if sum(weight) != 1:
                weight = [w / sum(weight) for w in weight]

        doc = nlp(sentence)
        sentence_token = [token.text.lower() for token in doc if token.is_alpha]
        score = 1
        
        ngram_list = {}
        for i in range(self.max_ngram):
            n = i + 1
            ngram_list[str(n)] = list(ngrams(pad_both_ends(sentence_token, n=n), n=n))

        for i, token in enumerate(sentence_token):
            if interpolation:
                interpolation_score = 0
                for j in range(self.max_ngram):
                    n = j + 1
                    pre_words = tuple(ngram_list[str(n)][i][:-1])
                    interpolation_score += weight[j] * self.score(token, pre_words)

                score *= interpolation_score

            elif backoff:
                backoff_score = 0
                for j in range(self.max_ngram):
                    n = j + 1
                    pre_words = tuple(ngram_list[str(n)][i][:-1])
                    if self.score(token, pre_words) != 0:
                        backoff_score = self.score(token, pre_words)
                        break

                score *= backoff_score  
  
        return score

    def vocab(self):
        return list(self.count[()].keys())
    
    def preprocess_input(self, predict_word, pre_words):
        if type(pre_words) == str:
            doc = nlp(pre_words)
            pre_words = tuple([token.text.lower() for token in doc if token.is_alpha])

        if type(pre_words) == list:
            pre_words = tuple([pre_word.lower() for pre_word in pre_words])
        
        assert type(pre_words) == tuple
        assert len(pre_words) <= self.max_ngram
        assert type(predict_word) == str

        predict_word = predict_word.lower()

        return predict_word, pre_words

        
    
lm = ProbabilisticLanguageModel(text)
