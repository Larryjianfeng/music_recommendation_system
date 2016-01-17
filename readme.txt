The code in process 
try both collaborative filtering and content based filtering method for recommendation system. 


The first code use a latent factor model, using randomized SVD parser. 
The second code will use a content based filtering method.

The second using the LDA model(latent dirichlet allocation) based on gensim package in python :
Subtracting the words count in the data first
using TF-IDF transformation to transform the corpus followed by step one. 
Train a LDA model to the corpus. 
I choose 20 topics to the model. 

The following is very interesting found in group 8 and group 9. 

"""


     songs labeled as 8 contains many of the following words, which is not contained in 9th group:
           will, death, god, blood, world, life, dark, soul 
     songs labeled as 9 contains many of the following words, which is not contained in 8th group:
           que, de, el,, la, en, mi, tu, se, los, yo, para, una


"""

Link for the data (millionsongs database): http://labrosa.ee.columbia.edu/millionsong/
Link for gensim: http://radimrehurek.com/gensim/
Link for lol: http://www.lolesports.com/en_US/