import os 
os.chdir('C:\Users\JianfengYan\Documents\GitHub\music_recommendation_system\data')
from scipy.sparse import csc_matrix

 

def read_songs():
	with open('kaggle_songs.txt','r') as f:
		songs = f.readlines()
	f.close()
	songs = [i.split()[0] for i in songs]
	return songs


def read_users():
	with open('kaggle_users.txt', 'r') as f:
		users = f.readlines()
	f.close()
	users = [i.split()[0] for i in users]
	return users

# there are about 400k songs, 110k users, 1450 relationships. 

def read_count():
	with open('kaggle_visible_evaluation_triplets.txt','r') as f:
		dics = f.readlines()
	dics = [i.strip().split('\t') for i in dics]
	return dics 

def make_data_set():

	songs = read_songs()
	dic_songs = {}
	for i in range(len(songs)):
		dic_songs[songs[i]] = i
	inv_songs = {v: k for k, v in dic_songs.items()}

	users = read_users()
	dic_users = {}
	for i in range(len(users)):
		dic_users[users[i]] = i
	inv_users = {v: k for k, v in dic_users.items()} 


	dics = read_count()
	rows = [dic_users[i[0]] for i in dics]
	cols = [dic_songs[i[1]] for i in dics]
	data = [int(i[2]) for i in dics]

	sp_matrix = csc_matrix((data, (rows, cols)), shape=(len(users), len(songs)))

	return [inv_users,inv_songs,sp_matrix,dic_songs,dic_users]






