import math
import numpy as np
from scipy import spatial


def entropy(predictions):
    ent = 0
    for ele in predictions:
        if ele > 0:
            ent += ele*math.log(ele, 10)
    return 0-ent


def similarity_numpy(x,y):
    x = np.array(x)
    y = np.array(y)

    result = np.matmul(np.matmul(x, np.identity(len(x))), y)
    print(len(result))
    return result.tolist()


def similarity(x,y):
    redun = 0
    for index, value in enumerate(x):
        redun += value * y[index]
    return redun


def max_similarity(value, list):
    index = 0
    max_redun = -100000
    for ele in list:
        # print(ele.tolist())
        redun = similarity(value,ele)
        if redun > max_redun:
            max_redun = redun
    return max_redun

def max_cosine_similarity(value, list):
    max_redun = -100000
    for ele in list:
        redun = 1-spatial.distance.cosine(value,ele)
        if redun > max_redun:
            max_redun = redun
    return max_redun
