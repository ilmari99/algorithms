def solution(S):
  """ Loop through string S, and check if the current element is correct, by
  checking what the previous element was.
  If the current element == previous, then insert the previous two elements.
  If the current element is different from expected and not equal to previous, then add the missing character.
  """
  next_map = {
    "a":"b",
    "b":"c",
    "c":"a",
  }
  if not S:
    return 0
  # Init vars
  curr = S[0]
  prev = ""
  inserts = 0
  i = 0
  # Loop until we can no more without errors
  while True:
    curr = S[i]
    # At the first char
    if prev == "":
      prev = S[i]
      i += 1
      continue
    # If the curr == prev, then two characters must be missing
    if curr == prev:
      S = S[:i] + next_map[prev]+next_map[next_map[prev]]+S[i:]
      #print("Inserted: {}".format(next_map[prev]+next_map[next_map[prev]]))
      inserts += 2
      i += 2
    # ELSE if curr != previous.next then add the missing character
    elif curr != next_map[prev]:
      S = S[:i] + next_map[prev] + S[i:]
      i += 1
      inserts += 1
    # Make prev the current, or latest added character
    prev = S[i]
    i += 1
    #print(S)
    if len(S) < i+1:
      break
  return inserts

if __name__ == "__main__":
    cases = {
        "aabcc" : 4,
        "abcabcabcabc" : 0,
        "aabbcc" : 6,
        "aaabbcb" : 7,
        "":0,
        "aaaaa":8,
        "abababab":3,
    }
    for c, a in cases.items():
        print("Case: {}".format(c))
        ans = solution(c)
        print("Answer: {}".format(ans))
        assert ans == a