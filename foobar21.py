'''
Returns the number of 'salutes' (2 / people meeting) from a string where '<' and '>' denote persons moving down an isle and their directions
For a custom string:
`python3 foobar21.py <string>`
For a demonstration, do not give a string argument
'''

def solution(s):
    """Returns the number of 'salutes' (2 / people meeting) from a string where '<' and '>' denote persons moving down an isle and their directions

    Args:
        s (str): a string with '<','-' or '>' characters that denote people moving or distance between them

    Returns:
        int: number of salutes
        
    Counts the salutes that minions make from left to right.
    Divides the string into parts where there is only one set of minions going right and one set going left
    for ex. for the stripped input string ">><>><<" the total salutes are
    salutes(">><") + salutes(">>>><<")
    """    
    s = s.replace("-","") #remove all dashes
    s = s.lstrip("<").rstrip(">")#remove all people who don't meet anyone
    char = s[0]
    i = 0
    substr = ""
    salutes = 0
    # Re
    while 1:
        # After stripping the string s always starts with ">"
        # The same is also true after for the substring
        if "<" not in substr and char == ">":
            substr = substr + ">"
        elif char == "<":
            substr = substr + "<"
        else:
            # Now substring contains the first sets of going right and left
            # for example: substr =  ">>><<", and actual string could be s = ">>><<>><><"
            # Hence we can remove the left most set from s after counting how many salutes they make
            # because the right most set going right is the last set of people they meet
            left = substr.count("<")
            right = substr.count(">")
            opposite = max([left,right])*min([left,right]) # Count how many salutes does the substring make
            salutes = salutes + 2*opposite
            if substr == s:
                break
            s = s.replace("<","",left) # Remove the left most set going left
            substr = ""
            i = -1
        i = i + 1
        try:
            char = s[i]
        except IndexError:
            char = None
    return salutes

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