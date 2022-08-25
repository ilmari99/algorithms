import numpy as np


def answer(path,exits,entrances):
    """ Turn a multisink-source flow problem, to single sink-source problem"""
    path = np.array(path)
    path = np.insert(path,0,[0 for i in range(len(path[0]))],axis=0)
    path = np.insert(path,len(path),[0 for i in range(len(path[0]))],axis=0)
    path = np.insert(path,len(path.T),[0 for i in range(len(path.T[0]))],axis=1)
    path = np.insert(path,0,[0 for i in range(len(path[0])+1)],axis=1)
    for ex in exits:
        path[ex+1][-1] = 10000000
    for ent in entrances:
        path[0][ent+1] = np.sum(path[ent+1])
    return path.tolist(),0,len(path)-1
    