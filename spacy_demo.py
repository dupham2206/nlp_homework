import spacy

# pip install spacy
# spacy download en_core_web_sm

nlp = spacy.load('en_core_web_sm')

# text = "this is an example of spacy."
text = "I can't believe it's already September. How the time flies!"

# Tokenize the text
doc = nlp(text)

# Print the tokens
print([token.text for token in doc])

# Print the idx of the tokens
print([token.idx for token in doc])

# Print sentences
print(list(doc.sents))

# Print is_alpha: whether the token consists of alphabetic characters or not
print([token.is_alpha for token in doc])

# Print is_stop: whether the token is a stop word or not
print([(token.is_stop, token.text) for token in doc])

# Print is_punct: whether the token is punctuation or not
print([(token.is_punct, token.text) for token in doc])

# spacy stop words
# print(nlp.Defaults.stop_words)

# remove stop words
print([token.text for token in doc if not token.is_stop])

# print lemma
print([token.lemma_ for token in doc])

# print pos
print([(token.text, token.pos_) for token in doc])

# print tag
print([(token.text, token.tag_) for token in doc])