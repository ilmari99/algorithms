from collections import defaultdict

PRESTATES = {
    1: (
        ((1,0),(0,0)),
        ((0,1),(0,0)),
        ((0,0),(1,0)),
        ((0,0),(0,1))
    ),
    0: (
        ((0, 1), (0, 1)),
        ((0, 1), (1, 1)),
        ((1, 0), (1, 1)),
        ((1, 1), (0, 0)),
        ((1, 1), (1, 0)),
        ((0, 1), (1, 0)),
        ((0, 0), (0, 0)),
        ((1, 0), (1, 0)),
        ((1, 0), (0, 1)),
        ((1, 1), (0, 1)),
        ((0, 0), (1, 1)),
        ((1, 1), (1, 1))
    )
}

def col_preimg_generator(c1, c2):
    for pre_c1 in c1:
        for pre_c2 in c2:
            if pre_c1[-1] == pre_c2[0]:
                yield tuple(pre_c1)+(pre_c2[1],)


def get_col_preimages(col):
    preimages = PRESTATES[col[0]]
    for _, cell in filter(lambda k: k[0]>0, enumerate(col)):
        preimages = col_preimg_generator(preimages, PRESTATES[cell])
    return tuple([tuple(zip(*pre)) for pre in preimages])


def solution(g):
    rot_g = tuple(zip(*g))
    preimages = defaultdict(int)
    for p in get_col_preimages(rot_g[0]):
        preimages[p[1]] += 1
    for _, col in filter(lambda k: k[0]>0, enumerate(rot_g)):
        nxt_preimages = {}
        for p in get_col_preimages(col):
            if p[0] in preimages:
                nxt_preimages[p[1]] = preimages[p[0]] + (nxt_preimages[p[1]] if p[1] in nxt_preimages else 0)
        preimages = nxt_preimages
    return sum(preimages.values())
