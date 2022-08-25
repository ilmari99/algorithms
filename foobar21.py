'''
Commander Lambda loves efficiency and hates anything
that wastes time. She's a busy lamb, after all! She
generously rewards henchmen who identify sources of inefficiency
and come up with ways to remove them. You've spotted one such 
source, and you think solving it will help you build the reputation you need to get promoted.

Every time the Commander's employees pass each other in the hall, 
each of them must stop and salute each other - one at a time - 
before resuming their path. A salute is five seconds long, so each 
exchange of salutes takes a full ten seconds (Commander Lambda's salute 
is a bit, er, involved). You think that by removing the salute requirement, 
you could save several collective hours of employee time per day. But first, 
you need to show her how bad the problem really is.

Write a program that counts how many salutes are exchanged during a typical 
walk along a hallway. The hall is represented by a string. For example: "--->-><-><-->-"

Each hallway string will contain three different types of characters: '>', 
an employee walking to the right; '<', an employee walking to the left; and '-', 
an empty space. Every employee walks at the same speed either to right or to the 
left, according to their direction. Whenever two employees cross, each of them salutes 
the other. They then continue walking until they reach the end, finally leaving the hallway. 
In the above example, they salute 10 times.

Write a function answer(s) which takes a string representing employees walking along 
a hallway and returns the number of times the employees will salute. s will contain at 
least 1 and at most 100 characters, each one of -, >, or <.

ME
===
At first I was like ohmygood and my initial solution was quite messy (but working).
Then a few days later, after talking to mom I realized how easy this was.


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
        print("State of the aisle:",aisle)
        salutes = solution(aisle)
        print("People currently in aisle perform",salutes, "salutes, assuming everyone who meets in the aisle perform 1 salute.")