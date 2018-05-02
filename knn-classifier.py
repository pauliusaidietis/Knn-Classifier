import numpy as np
import pandas as pd

df = pd.read_csv("1_mokymo_imtis.txt", index_col=0)

df_feat = pd.DataFrame(df, columns=df.columns[:-1])

k1 = 1
k2 = 3
predict = [6, 3]


def FindEuclidean(X, Y, data):
    squared = np.square(X - Y)
    summed = squared.sum(axis=1)
    root = np.sqrt(summed)
    withDist = pd.concat([data, root], axis=1)

    return withDist


def FindNEarest(k, EuclDist):
    EuclDist = EuclDist.sort_values([0])
    EuclDist.rename(columns={EuclDist.columns[3]: 'Distance'}, inplace=True)
    print(EuclDist)
    EuclDist = EuclDist.head(k)
    print(EuclDist)
    return EuclDist


def CountVote(targets):
    targets = targets.groupby('Klasė').count()
    targets = targets.drop(columns=['x2', 'Distance'])
    targets.rename(columns={'x1': 'voteCount'}, inplace=True)
    print("\n", targets)
    return targets


def MajorityVote(voteCount):
    if(len(voteCount.index) > 1):
        if voteCount.loc[0]['voteCount'] > voteCount.loc[1]['voteCount']:
            return 0
        elif voteCount.loc[0]['voteCount'] == voteCount.loc[1]['voteCount']:
            return "Both"
        else:
            return 1
    return voteCount.iloc[0]['voteCount']


distances = FindEuclidean(predict, df_feat, df)
targets = FindNEarest(k2, distances)
voteCount = CountVote(targets)


print("Object belongs to: ", MajorityVote(voteCount))
