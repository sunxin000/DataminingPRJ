import time
import datetime
import random
import numpy as np
from tabulate import tabulate
from surprise.model_selection import cross_validate
from surprise.model_selection import KFold
from surprise import Dataset
from surprise import NormalPredictor
from surprise import BaselineOnly
from surprise import KNNBasic
from surprise import KNNWithMeans
from surprise import KNNBaseline
from surprise import SVD
from surprise import SVDpp
from surprise import NMF
from surprise import SlopeOne
from surprise import CoClustering
from surprise import Reader
from tqdm import tqdm
from reading_in_data import user_rating_matrix
# SlopeOne, KNNBasic, KNNWithMeans, KNNBaseline, CoClustering, BaselineOnly
# The algorithms to cross-validate
classes = (SVD, NMF,  NormalPredictor)

np.random.seed(0)
random.seed(0)

R = user_rating_matrix[["userId", "movieId", "rating"]]
reader = Reader(rating_scale=(0.5, 5))
data = Dataset.load_from_df(
    R[['userId', 'movieId', 'rating']], reader)
kf = KFold(random_state=0)  # folds will be the same for all algorithms.

table = []
for klass in tqdm(classes):
    start = time.time()
    out = cross_validate(klass(), data, ['rmse', 'mae'], kf)
    cv_time = str(datetime.timedelta(seconds=int(time.time() - start)))
    mean_rmse = '{:.3f}'.format(np.mean(out['test_rmse']))
    mean_mae = '{:.3f}'.format(np.mean(out['test_mae']))
    new_line = [klass.__name__, mean_rmse, mean_mae, cv_time]
    print(tabulate([new_line], tablefmt="pipe"))  # print current algo perf
    table.append(new_line)

header = ['ml_least',
          'RMSE',
          'MAE',
          'Time'
          ]
print(tabulate(table, header, tablefmt="pipe"))