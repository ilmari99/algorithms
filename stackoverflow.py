#!/usr/bin/python3
from foobar41test import *
from multisink_source_wrapper_test import answer as multi_wrap
from scipy.optimize import linear_sum_assignment

if __name__ == "__main__":
        path = [
                [0,0,0,0,0,2,1,3],
                [0,0,0,0,0,4,0,0],
                [0,0,0,0,0,0,4,0],
                [0,0,0,0,0,0,2,0],
                [0,0,0,0,0,0,0,1],
                [0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0],
        ]
        sources = [0,1,2,3,4]
        sinks = [5,6,7]
        pathC = [[2,3,3],
                 [4,10,10],
                 [10,4,10],
                 [10,10,5]
        ]
        print(linear_sum_assignment(pathC,maximize=False))
        path,sources,sinks = multi_wrap(path,sinks,sources)
        for r in path:
                print(r)
        print(f"Sink: {sinks}")
        print(f"Sources: {sources}")
        flow_mat = solution(path,sinks,sources,debug=False)
        
        print(flow_mat)