import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
import joblib
import random
from reading_in_data import movies_df, user_rating, ratings_pivot


if __name__ == "__main__":
    model = NMF(
        n_components=20,
        init='random',
        random_state=10,
        max_iter=1000
    )
    model.fit(ratings_pivot)
    print(model.reconstruction_err_)
    joblib.dump(model, "nmf.sav")

    P = model.transform(ratings_pivot)
    print(P.shape)
    Q = model.components_.T
    print(Q.shape)
    new_user = {1: 5, 50: 4}
    new_user = pd.DataFrame(
        new_user, index=[random.randint(1, 610)], columns=ratings_pivot.columns)
    new_user.fillna(3, inplace=True)
    P_new_user = model.transform(new_user)
    user_pred = pd.DataFrame(np.dot(P_new_user, Q.T), columns=ratings_pivot.columns,
                             index=[new_user.index.unique()])

    print(user_pred)
