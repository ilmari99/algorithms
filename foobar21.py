'''
Returns the number of 'salutes' (2 / people meeting) from a string where '<' and '>' denote persons moving down an isle and their directions
For a custom string:
`python3 foobar21.py <string>`
For a demonstration, do not give a string argument
'''

def solution(s : str):
    s.replace("-","")
    count = 0
    for i,c in enumerate(s[:len(s)]):
        if c == ">":
            for c2 in s[i+1:]:
                if c2 == "<":
                    count += 1
    return 2*count
                    

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:# If no cmd line argument is provided
        ss = ["<<>><",">----<",">>>><<<<",">><<--->-",">>><<","-<-<-<-<->>-<<><<-->>--<"]
        for s in ss:
            ans = solution(s)
            print(s,ans)
    else:
        aisle = sys.argv[1]
        print(f"State of the aisle: {aisle}")
        salutes = solution(aisle)
        print(f"People currently in aisle perform {salutes} salutes, assuming everyone who meets in the aisle perform 1 salute.")