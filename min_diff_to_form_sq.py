from collections import Counter
def solution(A):
  """ Count the occurrences of the values and loop the constructed counter in two loops.
  If there are two same size elements, find another pair. Store difference, and continue looping.
  """
  # Create a counter containing length : occurrences -pairs
  counter = Counter(A)
  # Initialize vars
  min_diff = float("inf")
  sol = None  # incase we want the solution
  # Loop through the counter, and see if there are atleast 2 elements of the same length
  for s1,count in counter.items():
    if count < 2:
      continue
    # If there are 2 elements, try to find a new set of two elements
    for s2, count in counter.items():
      if count < 2 or (s2 == s1 and count < 4):
        continue
      diff = abs(s1-s2)
      # Store as smallest diff if applicable
      if diff < min_diff:
        min_diff = diff
        sol = (s1,s2)
        
  return min_diff if sol else -1

if __name__ == "__main__":
    cases = {
        (911, 1, 3, 1000, 1000, 2, 2, 999, 1000, 911) : 89,
         (4, 1, 1, 1, 3) : -1,
        (2,2,2,2,2) : 0,
        (33,33,12,12,3,11,5,5) : 7,
        (1,):-1,
        tuple():-1,
    }
    for c, a in cases.items():
        print("Case: {}".format(c))
        ans = solution(c)
        print("Answer: {}".format(ans))
        assert ans == a