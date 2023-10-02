"""
UET - Natural Language Processing - 2023-2024 - Semester 1 - Assignment 1
20020039
Pham Tien Du
"""

import spacy
from nltk.util import ngrams
from nltk.lm.preprocessing import pad_both_ends

nlp = spacy.load('en_core_web_sm')
text = "The boy likes books. The books are bank books. Some books are on the table. Some boys books the table. The book has many likes. Many banks like books."

class ProbabilisticLanguageModel():
    def __init__(self, data, max_ngram=3, smoothing=0.0):
        """Init Probabilistic Language Model (N-gram)

        Args:
            data (str): paragraph of text for training
            max_ngram (int, optional): max gram for N-gram to preprocessing data and using. Defaults to 3.
            smoothing (float, optional): smoothing number for Laplace smoothing. Defaults to 0.
        """

        self.data = data
        self.max_ngram = max_ngram
        self.smoothing = smoothing

        # get count and probability for infer N-gram model
        self.count, self.probability = self.preprocess_data(data)

    def preprocess_data(self, data):
        """Preprocess data for N-gram model

        Args:
            data (str): paragraph of text for training.

        Returns:
            count, probability: (dict, dict) count and probability for N-gram model.
        """

        # init ngram_list.
        # ngram_list['1'] is uni-gram, ngram_list['2'] is bi-gram, ngram_list['3'] is tri-gram...
        ngram_list = {}
        for n in range(1, self.max_ngram + 1):
            ngram_list[str(n)] = []

        # get doc from spacy nlp
        doc = nlp(data)

        # for each sentence in doc, with each n, get elements from tokens and append to ngram_list[n]
        for sentence in doc.sents:

            # get token from sentence
            sentence_token = [token.text.lower() for token in sentence if token.is_alpha]

            # for each n, get elements from tokens and append to ngram_list[n]
            # for example, with 2 sentences "The boy likes books. The books are bank books." and bi-gram we have:
            # ngram_list['2']: [('<s>', 'the'), ('the', 'boy'), ('boy', 'likes'), ('likes', 'books'), ('books', '</s>'), 
            # ('<s>', 'the'), ('the', 'books'), ('books', 'are'), ('are', 'bank'), ('bank', 'books'), ('books', '</s>')]

            # pad_both_ends with padding='<s>' and '</s>' for each sentence depend on n
            # ngrams() will get tuples of n-gram from tokens

            for n in range(1, self.max_ngram + 1):
                ngram_list[str(n)].extend(list(ngrams(pad_both_ends(sentence_token, n=n), n=n)))

        # ngram_list only list of tuples, we need to count and get probability for each n-gram
        # count[context][word] is count of word in context
        # sum[context] is sum of count of all word in context
        count = {}
        sum = {}
        
        # With each n in ngram_list, we count and get sum for each n-gram
        for n in range(1, self.max_ngram + 1):
            for element in ngram_list[str(n)]:

                # element[:-1] is context, element[-1] is word
                # for example, ('<s>', 'the', 'books'). element[:-1] = ('<s>', 'the'), element[-1] = 'books'
                context = element[:-1]
                word = element[-1]

                # if context not in count, init count[context] = {}
                if context not in count:
                    count[context] = {}
                    sum[context] = 0

                # if word not in count[context], init count[context][word] = 0
                if word not in count[context]:
                    count[context][word] = 0
                
                # plus 1 for count[context][word] and sum[context]
                count[context][word] += 1
                sum[context] += 1
        
        # get len_vocab for Laplace smoothing
        self.len_vocab = len(count[()])
        self.vocab_list = list(count[()].keys())
        self.sum = sum

        # get probability from count and sum
        probability = {}
        for context in count:
            probability[context] = {}

            # for each word in context, get probability
            for predict_word in count[context]:
                
                # add smoothing for Laplace smoothing. If smoothing = 0, it is MLE
                probability[context][predict_word] = (count[context][predict_word] + self.smoothing) / (sum[context] + self.smoothing * self.len_vocab)

        # return count and probability
        return count, probability

    def count_gram(self, predict_word, context):
        """Count number of predict_word in context in training data

        Args:
            predict_word (str): predict word
            context (tuple or list or str): context of predict word

        Returns:
            count: number of predict_word in context in training data
        """
        predict_word, context = self.preprocess_input(predict_word, context)
        if context not in self.count:
            return 0
        if predict_word not in self.count[context]:
            return 0
        return self.count[context][predict_word]

    def score(self, predict_word, context):
        """Get probability of predict_word in context

        Args:
            predict_word (str): predict word
            context (tuple or list or str): context of predict word

        Returns:
            score: probability of predict_word in context
        """
        predict_word, context = self.preprocess_input(predict_word, context)

        # If predict_word not in vocabulary, return 0
        if predict_word not in self.vocab_list and predict_word != '</s>' and predict_word != '<s>':
            return 0

        # If context not in training data, return 1 / len_vocab = smoothing / (len_vocab * smoothing)
        if context not in self.count:
            return 1 / self.len_vocab if self.smoothing != 0.0 else 0

        # If predict_word not in context, return smoothing / (sum[context] + smoothing * len_vocab)
        if predict_word not in self.count[context]:
            return self.smoothing / (self.sum[context] + self.smoothing * self.len_vocab)
        
        # Else, return probability[context][predict_word]
        return self.probability[context][predict_word]
    
    def score_sentence(self, sentence, n_choose=2, interpolation=False, backoff=False, weight=[1, 1, 1]):
        """Get probability of sentence

        Args:
            sentence (str): sentence for get score
            n_choose (int, optional): n-gram for get score. Defaults to 2. Only use when interpolation=False and backoff=False.
            interpolation (bool, optional): interpolation mode. Defaults to True.
            backoff (bool, optional): backoff mode. Defaults to False.
            weight (list, optional): weight for interpolation mode. Defaults to [1, 1, 1].

        Returns:
            _type_: _description_
        """
        assert not (interpolation and backoff), "interpolation and backoff can't be True at the same time"
        
        if not interpolation and not backoff:
            assert n_choose <= self.max_ngram, f"n_choose must be <= {self.max_ngram}"

        # Normalize weight
        if interpolation:
            assert len(weight) == self.max_ngram
            if sum(weight) != 1:
                weight = [w / sum(weight) for w in weight]

        # Preprocess sentence
        doc = nlp(sentence)
        sentence_token = [token.text.lower() for token in doc if token.is_alpha]
        score = 1
        
        # Get n-gram list from sentence
        ngram_list = {}
        for n in range(1, self.max_ngram + 1):
            ngram_list[str(n)] = list(ngrams(pad_both_ends(sentence_token, n=n), n=n))

        # Get score from n-gram list
        for i, token in enumerate(sentence_token):

            # interpolation mode
            if interpolation:
                interpolation_score = 0

                # for each n, get context, add score with weight
                for n in range(1, self.max_ngram + 1):
                    context = tuple(ngram_list[str(n)][i][:-1])
                    interpolation_score += weight[n - 1] * self.score(token, context)

                score *= interpolation_score

            # backoff mode
            # backoff should be used with smoothing = 0
            elif backoff:
                backoff_score = 0

                # for each n, get context and score until score != 0 
                for j in range(self.max_ngram):
                    n = j + 1
                    context = tuple(ngram_list[str(n)][i][:-1])
                    if self.score(token, context) != 0:
                        backoff_score = self.score(token, context)
                        break

                score *= backoff_score 

        # single n-gram mode
        if not interpolation and not backoff:
            for element in ngram_list[str(n_choose)]:
                context = element[:-1]
                predict_word = element[-1]
                score *= self.score(predict_word, context)
  
        return score

    def vocab(self):
        """Get vocabulary of training data"""
        return list(self.count[()].keys())
    
    def preprocess_input(self, predict_word, context):
        """Preprocess input for count_gram and score function

        Args:
            predict_word (str): predict word
            context (tuple or list or str): context of predict word

        Returns:
            predict_word, context (tuple): predict word and context after preprocess
        """

        # if context is str, preprocess context with spacy nlp
        if type(context) == str:
            doc = nlp(context)
            context = tuple([token.text.lower() for token in doc if token.is_alpha])

        # if context is list or tuple, preprocess context with lower()
        if type(context) == list or type(context) == tuple:
            context = tuple([pre_word.lower() for pre_word in context])

        assert type(context) == tuple
        assert len(context) <= self.max_ngram
        assert type(predict_word) == str

        # preprocess predict_word with lower()
        predict_word = predict_word.lower()

        return predict_word, context


if __name__ == "__main__":
    print("Init Probabilistic Language Model with smoothing = 2.0")
    lm = ProbabilisticLanguageModel(text, max_ngram=3, smoothing=2.0)
    print(f"Count 'are' after context 'books': {lm.count_gram('are', ('books',))}")
    print(f"Probability 'are' after context 'books': {lm.score('are', ('books',))}")

    print(f"Count 'are' after context 'the books': {lm.count_gram('are', ('the', 'books'))}")
    print(f"Probability 'are' after context 'the books': {lm.score('are', ('the', 'books'))}")

    print(f"Count 'are' after context 'the books are': {lm.count_gram('are', ('the', 'books', 'are'))}")
    print(f"Probability 'are' after context 'the books are': {lm.score('are', ('the', 'books', 'are'))}")

    print(f"Probability of sentence 'the books are books' with bigram: {lm.score_sentence('the books are books', n_choose=2)}")
    print(f"Probability of sentence 'the books are books' with unigram: {lm.score_sentence('the books are books', n_choose=1)}")
    print(f"Probability of sentence 'the books are books' with interpolation: {lm.score_sentence('the books are books', interpolation=True, weight=[0.5, 0.3, 0.2])}")

    print(f"Vocabulary: {lm.vocab()}")

    print("Init Probabilistic Language Model with smoothing = 0.0")
    lm = ProbabilisticLanguageModel(text, max_ngram=3, smoothing=0.0)
    print(f"Probability of sentence 'the books are books' with backoff: {lm.score_sentence('the books are books', backoff=True)}")