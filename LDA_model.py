import sys
sys.path.insert(0, 'C:\Users\JianfengYan\Documents\GitHub\music_recommendation_system')
import os 
os.chdir('C:\Users\JianfengYan\Documents\GitHub\music_recommendation_system\data')
import data_manipulation
import gensim
import numpy as np

make_dataset = data_manipulation.make_data_set()
inv_songs = make_dataset[1]
inv_users = make_dataset[0]
sp_matrix = make_dataset[2]
dic_songs = make_dataset[3]
dic_users = make_dataset[4]


with open('mxm_dataset_test.txt') as f:
	g = f.readlines()
f.close()

dic_words = g[17].strip().split(',')
dic_words = np.array(dic_words)

with open('mxm_dataset_train.txt') as f:
	gg = f.readlines()
f.close()

corpus = []


track_list = []
for i in range(18, len(g)):
	item = g[i]
	corp = []
	items = item.strip().split(',')
	track_list.append(items[0])
	for j in range(2,len(items)):
		word_count = items[j].split(':')
		corp.append((int(word_count[0]), int(word_count[1])))

	corpus.append(corp)


tfidf = gensim.models.TfidfModel(corpus)
corpus_tf = [tfidf[i] for i in corpus]

lda = gensim.models.LdaModel(corpus_tf, num_topics = 20, iterations=300)

topics_of_tracks = []
for i in range(len(corpus_tf)):
	if i%100 == 0 :
		print str(i) + ' has been processed'
	prob = lda[corpus_tf[i]]
	agent = 0
	for j in prob: 
		if j[1] > agent:
			agent = j[1]
			topic = j[0]
	topics_of_tracks.append(topic)

topics_of_tracks = np.array(topics_of_tracks)

def make_count(n):
	count = 0 
	topic_list = []
	for i in range(len(corpus)):
		if topics_of_tracks[i] == n:
			topic_list.append(i)
	top_words = [0]*5000
	for i in topic_list:
		corpu = corpus_tf[i]
		for cor in corpu:
			top_words[cor[0]-1] = top_words[cor[0]-1] + cor[1]
	top_words = np.array(top_words)
	top_words = top_words/sum(top_words)
	return top_words

def make_comparing_data(m,n):
	a = make_count(m)
	b = make_count(n)
	return pd.DataFrame({'words': dic_words, 'diff_ratio':(a-b)}).sort('diff_ratio')


# data should like [['track_name',[23:3, 33:4...]],]

da = make_comparing_data(8,9)



"""

     in the end, a very interesting fact: 
     I find that songs labeled as 8 contains many of the following words, which is not contained in 9th group:
           will, death, god, blood, world, life, dark, soul 
     9 contains many of the following words, which is not contained in 8th group:
           que, de, el,, la, en, mi, tu, se, los, yo, para, una


"""