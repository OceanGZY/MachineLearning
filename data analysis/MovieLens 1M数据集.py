# coding:utf-8
'''
    auth:gzy
    date:2018/12/11
'''

import pandas as pd
unames = ['user_id','gender','age','occupation','zip']
users = pd.read_table('/Users/gaozhiyong/Documents/GitHub/MachineLearning/data analysis/database/datasets/movielens/users.dat',sep='::',header=None,names=unames)

rnames =['user_id','movie_id','rating','timestamp']
ratings = pd.read_table('/Users/gaozhiyong/Documents/GitHub/MachineLearning/data analysis/database/datasets/movielens/ratings.dat',sep='::',header=None,names=rnames)

mnames =['movie_id','title','genres']
movies = pd.read_table('/Users/gaozhiyong/Documents/GitHub/MachineLearning/data analysis/database/datasets/movielens/movies.dat',sep='::',header=None,names=mnames)

print(users[:5])

print(ratings[:5])

print(movies[:5])