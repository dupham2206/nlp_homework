import re

txt = "The rain in Spain"

# Find ...
print(re.search("^The.*Spain$", txt))

# Find all "in" in the string
print(re.findall("in", txt))

# Split all string with white-space character
print(re.split("\s", txt))

# Split the string with white-space character only at the first 2 occurrences
print(re.split("\s", txt, 2))

# Replace the first 2 occurrences of white-space character with "_"
print(re.sub("\s", "_", txt, 2))

import nltk
from nltk.tokenize import word_tokenize, TreebankWordTokenizer, RegexpTokenizer

# nltk.download('punkt')

text = "I can't believe it's already September. How the time flies!"

# Tokenize the text by words
print(word_tokenize(text))

# Tokenize the text by tree bank tokenizer
tree_bank_tokenizer = TreebankWordTokenizer()
print(tree_bank_tokenizer.tokenize(text))

# Tokenize the text by regular expression tokenizer
regexp_tokenizer = RegexpTokenizer(r"\w+")
print(regexp_tokenizer.tokenize(text))


from nltk.stem import PorterStemmer, WordNetLemmatizer

words = ["game", "gaming", "gamed", "games", "are", "be", "being", "been", "is"]

# nltk.download('wordnet')
# nltk.download('omw-1.4')

# Stem the words
porter_stemmer = PorterStemmer()
print([porter_stemmer.stem(word) for word in words])

lemmentizer = WordNetLemmatizer()
print([lemmentizer.lemmatize(word, pos="v") for word in words])


