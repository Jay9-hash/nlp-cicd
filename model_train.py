# Regular Expression
import re
import joblib
import nltk

# Download required NLTK data
nltk.download('punkt')
nltk.download('punkt_tab') 
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Training Data
texts = [
    "I love this product",
    "This product is very bad",
    "Amazing quality and good service",
    "Worst experience ever"
]

labels = [1, 0, 1, 0]

# Step 1: Cleaning
cleaned_texts = []

for sentence in texts:
    sentence = sentence.lower()
    sentence = re.sub(r'[^a-z\s]', '', sentence)
    cleaned_texts.append(sentence)

# Step 2: Stopwords & Lemmatization
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

final_texts = []

for sentence in cleaned_texts:
    tokens = word_tokenize(sentence)
    words = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    final_texts.append(" ".join(words))

# Step 3: Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(final_texts)

# Step 4: Model Training
model = LogisticRegression()
model.fit(X, labels)

# Step 5: Save Model & Vectorizer
joblib.dump(model, 'my_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print(" Model Ready for Prediction")





#=============================================================

# # re -Reular Expression
# #Used to clean the Text
# #"Hell !!! image"  ---"hello"
# #re = cleaning the noise data
# import re 
# #joblib helps us to build and compile the all libraries to package
# #1000 lines training predictions -->my_model.pkl -->Flask. --> Github-->Docker--> Jenkins-->AWS
# #used to "save model" to "file"
# #After Training, we dont want to repeat again and again 
# #jobib for CI/CD pipeline
# import joblib

# # import nltk
# # nlp library
# # punkit -- sentence splitter

# import nltk
# nltk.download('punkt')
# nltk.download('punkt_tab')

# #identify stop words
# # nltk.downlaod ('stopwords')
# nltk.download('stopwords')  


# #English Dictionary
# nltk.download('wordnet')

# #I love NLP
# #convert to upper case to lower case = i love nlp
# #["i","love","nlp"]
# # nltk use for to clean the model
# #for mackbook python -m venv venev
# # source venv/bin/activate
# #pip install nltk

# from nltk.tokenize import word_tokenize

# # removing to the meaning less words EX : i love nlp Remove i
# # identifies the stopwords
# # is, am , the, a, an , was, were
# from nltk.corpus import stopwords

# #wordNetLematizer = This syntex gives the meaning full sentence
# # converts words to dictionary root
# #playing - play
# #better --> good
# # awesome --> good
# #hate --> bad
# # from nltk.stem import wordNetLematizer
# from nltk.stem import WordNetLemmatizer  


# # convert your word to text --numbers
# # meaningfull words -- high priority
# # lessfull words -- less priority

# from sklearn.feature_extraction.text import TfidVectorizer

# # logisticRegression -- is the ML Algorithm
# # Output : 0 (0r). 1
# from sklearn.linear_model import LogisticRegression

# #Training Data
# texts = ["I love this product",
#          "This product is very bad",
#          "Amazing quality and good service",
#          "Worst experience ever"
#          ]

# labels = [1,0,1,0]

# #Step : Cleaning Process

# cleaned_texts = []
# for sentence in texts:
#     sentence = sentence.lower()
#     # re.sub(r'[^a-z\s]','',sentence)
#     sentence = re.sub(r'[^a-z\s]', '', sentence)  
#     cleaned_texts.append(sentence)

#     print(cleaned_texts)


# stop_words =set(stopwords.words('english'))
# lemmatizer = wordNetLematizer()

# final_words = []
# final_texts.append(...)

# for sentence in cleaned_texts:
#     tokens = word_tokenize(sentence)
#     # words = [lemmatizer.lemmatize(word)] for word in tokens if word not in stop_words
#     words = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

#     final_texts.append("".join(words))
#     print(final_texts)

#     vectorizer = TfidVectorizer()
#     vectorizer.fit_transform(final_texts)

#     model = LogisticRegression ()
#     # model.fit(final_texts,lables)
#     x= vectorizer.fit_transform(final_texts)
#     model.fit(x,labels)

#     joblib.dump(model,'my_model.pkl')
#     joblib.dump(vectorizer,'vectorizer.pkl')

#     print("Model Ready for Prediction")

# =================================================================



