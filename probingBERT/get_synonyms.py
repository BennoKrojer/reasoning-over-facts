from nltk.corpus import wordnet
from nltk.corpus import stopwords
import nltk
import pandas as pd

nltk.download('wordnet')
df = pd.read_csv('unigram_freq.csv')
with open('synonyms','w') as file:
    for index, row in df.iterrows():
        if index < 5000:
            try:
                word = row['word']
                synsets = wordnet.synsets(word)
                if word not in stopwords.words('english'):
                    for synset in synsets:
                        file.write(str([lemma.name() for lemma in synset.lemmas()])+'\n')
            except AttributeError:
                continue
