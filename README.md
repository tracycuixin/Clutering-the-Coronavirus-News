# Clutering-the-Coronavirus-News
In this project, I am doing clutering on coronavirus global news to helps users keep up with the ongoing pandemic issues and search the specific topics.

The dataset comes from CBC news's own search result regarding coronavirus.


Firstly, I would process the text from the body of news using Natural Language Processing (NLP).

Secondly, I would use Principal Component Analysis (PCA) to project down the dimensions of data and use t-SNE to reduce dimensionality, so that we can visualize clusters of instances in high-dimensional space.

Then, I would apply k-means clustering on data and apply Latent Dirichlet Allocation (LDA) to discover keywords from each cluster.

At the end, I investigate the clusters with classification using Stochastic Gradient Descent (SGD).

# Requirements
`numpy`: For fast matrix operations.

`pandas`: For analysing and getting insights from datasets.

`seaborn` : For enhancing the style of matplotlib plots.

`matplotlib`: For creating graphs and plots.

# Data explore

## Length of News
We use `pd.Series` to get the length of the News here
```
lengths = pd.Series([len(x) for x in df.text])
```

# Some feature engineering
I count the words in the description and the text of each news.
```
df['description_word_count'] = df['description'].apply(lambda x: len(x.strip().split()))  # word count in abstract
df['text_word_count'] = df['text'].apply(lambda x: len(x.strip().split()))  # word count in body
```

# Handle Possible Duplicates
We don't need duplicate news here, so I use `drop_duplicates` to get rid of duplicates
```
df.drop_duplicates(['description', 'text'], inplace=True)
```

# Handling multiple languages
We will use only English news here. So in case of the news have different mulpiple languages, I use `langdetect` to detect what kinds of languages do they have.

## Requirements
```
from tqdm import tqdm
from langdetect import detect
from langdetect import DetectorFactory
```

## Go through the data
```
# set seed
DetectorFactory.seed = 0

# hold label - language
languages = []

# go through each text
for ii in tqdm(range(0,len(df))):
    # split by space into list, take the first x intex, join with space
    text = df.iloc[ii]['text'].split(" ")
    
    lang = "en"
    try:
        if len(text) > 50:
            lang = detect(" ".join(text[:50]))
        elif len(text) > 0:
            lang = detect(" ".join(text[:len(text)]))
    except Exception as e:
        all_words = set(text)
        try:
            lang = detect(" ".join(all_words))
        except Exception as e:
            
            try:
                lang = detect(df.iloc[ii]['description'])
            except Exception as e:
                lang = "unknown"
                pass
    
    # get the language    
    languages.append(lang)
```

And now, let's see how many languages the news have.
```
from pprint import pprint

languages_dict = {}
for lang in set(languages):
    languages_dict[lang] = languages.count(lang)
    
print("Total: {}\n".format(len(languages)))
pprint(languages_dict)
```

# NLP
Now it's time for the NLP part.
First of all, download the spacy bio parser.
```
from IPython.utils import io
with io.capture_output() as captured:
    !pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_lg-0.2.4.tar.gz
```

## Requirements
```
#NLP 
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import en_core_sci_lg  # model downloaded in previous step
```

## Stopwords
Stopwords are common words that will act as noise in the clustering step.
It is part of the preprocessing will be finding and removing stopwords.
```
import string

punctuations = string.punctuation
stopwords = list(STOP_WORDS)
```
Next, I create a function that will process the text data for us.

For this purpose, I will be using the spacy library. This function will convert text to lower case, remove punctuation, and find and remove stopwords. For the parser, we will use `en_core_sci_lg`. This is a model for processing biomedical, scientific or clinical text.
```
# Parser
parser = en_core_sci_lg.load(disable=["tagger", "ner"])
parser.max_length = 7000000

def spacy_tokenizer(sentence):
    mytokens = parser(sentence)
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
    mytokens = [ word for word in mytokens if word not in stopwords and word not in punctuations ]
    mytokens = " ".join([i for i in mytokens])
    return mytokens

tqdm.pandas()
df["processed_text"] = df["text"].progress_apply(spacy_tokenizer)
```

# Vectorization
Now that we have pre-processed the data, it is time to convert it into a format that can be handled by our algorithms. For this purpose we will be using `tf-idf`. This will convert our string formatted data into a measure of how important each word is to the instance out of the literature as a whole.
```
from sklearn.feature_extraction.text import TfidfVectorizer
def vectorize(text, maxx_features):
    
    vectorizer = TfidfVectorizer(max_features=maxx_features)
    X = vectorizer.fit_transform(text)
    return X
```

We will be clustering based off the content of the body text. The maximum number of features will be limited. Only the top 2 ** 12 features will be used, essentially acting as a noise filter. Additionally, more features cause painfully long runtimes.
```
text = df['processed_text'].values
X = vectorize(text, 2 ** 12)
X.shape
```

# PCA & Clustering
I will apply Principle Component Analysis (PCA) to our vectorized data. The reason for this is that by keeping a large number of dimensions with PCA, you don’t destroy much of the information, but hopefully will remove some noise/outliers from the data, and make the clustering problem easier for k-means.

```
from sklearn.decomposition import PCA

pca = PCA(n_components=0.95, random_state=42)
X_reduced= pca.fit_transform(X.toarray())
X_reduced.shape
```
To separate the literature, k-means will be run on the vectorized text. Given the number of clusters, k, k-means will categorize each vector by taking the mean distance to a randomly initialized centroid. The centroids are updated iteratively.

```
from sklearn.cluster import KMeans
```
To find the best k value for k-means we'll look at the distortion at different k values. Distortion computes the sum of squared distances from each point to its assigned center. When distortion is plotted against k there will be a k value after which decreases in distortion are minimal. This is the desired number of clusters.

```
from sklearn import metrics
from scipy.spatial.distance import cdist

# run kmeans with many different k
distortions = []
K = range(2, 50)
for k in K:
    k_means = KMeans(n_clusters=k, random_state=42).fit(X_reduced)
    k_means.fit(X_reduced)
    distortions.append(sum(np.min(cdist(X_reduced, k_means.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
```
## Run k-means

Now that we have an appropriate k value, we can run k-means on the PCA-processed feature vector (X_reduced).

```
k = 20
kmeans = KMeans(n_clusters=k, random_state=42)
y_pred = kmeans.fit_predict(X_reduced)
df['y'] = y_pred
```
# Dimensionality Reduction with t-SNE
Using t-SNE we can reduce our high dimensional features vector to 2 dimensions. By using the 2 dimensions as x,y coordinates, the body_text can be plotted.

t-Distributed Stochastic Neighbor Embedding (t-SNE) reduces dimensionality while trying to keep similar instances close and dissimilar instances apart. It is mostly used for visualization, in particular to visualize

## clusters of instances in high-dimensional space
```
from sklearn.manifold import TSNE

tsne = TSNE(verbose=1, perplexity=100, random_state=42)
X_embedded = tsne.fit_transform(X.toarray())
```

# Topic Modeling on Each Cluster
For topic modeling, we will use LDA (Latent Dirichlet Allocation). In LDA, each document can be described by a distribution of topics and each topic can be described by a distribution of words.
```
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
```

First we will create 20 vectorizers, one for each of our cluster labels
```
vectorizers = []
    
for ii in range(0, 20):
    # Creating a vectorizer
    vectorizers.append(CountVectorizer(min_df=5, max_df=0.9, stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}'))
```
Now we will vectorize the data from each of our clusters.
```
vectorized_data = []

for current_cluster, cvec in enumerate(vectorizers):
    try:
        vectorized_data.append(cvec.fit_transform(df.loc[df['y'] == current_cluster, 'processed_text']))
    except Exception as e:
        print("Not enough instances in cluster: " + str(current_cluster))
        vectorized_data.append(None)
```
Topic modeling will be performed through the use of Latent Dirichlet Allocation (LDA). This is a generative statistical model that allows sets of words to be explained by a shared topic

```
# number of topics per cluster
NUM_TOPICS_PER_CLUSTER = 20

lda_models = []
for ii in range(0, 20):
    # Latent Dirichlet Allocation Model
    lda = LatentDirichletAllocation(n_components=NUM_TOPICS_PER_CLUSTER, max_iter=10, learning_method='online',verbose=False, random_state=42)
    lda_models.append(lda)

```
For each cluster, we had created a correspoding LDA model in the previous step. We will now fit_transform all the LDA models on their respective cluster vectors
```
clusters_lda_data = []

for current_cluster, lda in enumerate(lda_models):
    # print("Current Cluster: " + str(current_cluster))
    
    if vectorized_data[current_cluster] != None:
        clusters_lda_data.append((lda.fit_transform(vectorized_data[current_cluster])))
 ```
 Extracts the keywords from each cluster
 ```
 # Functions for printing keywords for each topic
def selected_topics(model, vectorizer, top_n=3):
    current_words = []
    keywords = []
    
    for idx, topic in enumerate(model.components_):
        words = [(vectorizer.get_feature_names()[i], topic[i]) for i in topic.argsort()[:-top_n - 1:-1]]
        for word in words:
            if word[0] not in current_words:
                keywords.append(word)
                current_words.append(word[0])
                
    keywords.sort(key = lambda x: x[1])  
    keywords.reverse()
    return_values = []
    for ii in keywords:
        return_values.append(ii[0])
    return return_values
 ```
Append list of keywords for a single cluster to 2D list of length NUM_TOPICS_PER_CLUSTER
```
all_keywords = []
for current_vectorizer, lda in enumerate(lda_models):
    # print("Current Cluster: " + str(current_vectorizer))

    if vectorized_data[current_vectorizer] != None:
        all_keywords.append(selected_topics(lda, vectorizers[current_vectorizer]))
```

