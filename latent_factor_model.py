# see more description for this class in 
# http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html#sklearn.decomposition.TruncatedSVD.fit

from sklearn.decomposition import TruncatedSVD
import os 
os.chdir('...')
import data_manipulation.py


make_dataset = data_manipulation.make_data_set()
inv_songs = make_dataset[1]
inv_users = make_dataset[0]
sp_matrix = make_dataset[2]



G = TruncatedSVD(n_components = 30, 
	n_iter = 40, 
	tol = 0.1,
	random_state=42)


G.fit(sp_matrix)

# s is the listening data of a user, with len(inv_songs) columns, most of it is 0
def make_recommendation(G,s): 
	score = G.transform(s)
	inv_score = G.inverse_transform(score)[0]
	res = []

	for i in range(len(inv_score)):
		if inv_score[i] > 0.1:
			res.append([i,inv_score[i]])
	reco_list = []

	for re in res: 
		if s[re[0]] == 0:
			reco_list.append(inv_songs[re[0]])
	return reco_list

#s = [10]*100 + [0]*(len(inv_songs) - 100)
#print make_recommendation(G, s)




