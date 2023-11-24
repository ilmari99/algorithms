import multiprocessing
import os
import time

"""
The preparations are finally complete; you and the Elves leave camp on foot and begin to make your way toward the star fruit grove.

As you move through the dense undergrowth, one of the Elves gives you a handheld device. He says that it has many fancy features, but the most important one to set up right now is the communication system.

However, because he's heard you have significant experience dealing with signal-based systems, he convinced the other Elves that it would be okay to give you their one malfunctioning device - surely you'll have no problem fixing it.

As if inspired by comedic timing, the device emits a few colorful sparks.

To be able to communicate with the Elves, the device needs to lock on to their signal. The signal is a series of seemingly-random characters that the device receives one at a time.

To fix the communication system, you need to add a subroutine to the device that detects a start-of-packet marker in the datastream. In the protocol being used by the Elves, the start of a packet is indicated by a sequence of four characters that are all different.

The device will send your subroutine a datastream buffer (your puzzle input);

---
your subroutine needs to identify the first position where the four most recently received characters were all different.
Specifically, it needs to report the number of characters from the beginning of the buffer to the end of the first such four-character marker.
---

For example, suppose you receive the following datastream buffer:

mjqjpqmgbljsphdztnvjfqwrcgsmlb
After the first three characters (mjq) have been received, there haven't been enough characters received yet to find the marker. The first time a marker could occur is after the fourth character is received, making the most recent four characters mjqj. Because j is repeated, this isn't a marker.

The first time a marker appears is after the seventh character arrives. Once it does, the last four characters received are jpqm, which are all different. In this case, your subroutine should report the value 7, because the first start-of-packet marker is complete after 7 characters have been processed.

Here are a few more examples:

bvwbjplbgvbhsrlpgdmjqwftvncz: first marker after character 5
nppdvjthqldpwncqszvftbrmjlhg: first marker after character 6
nznrnfrfntjfmvfwmzdfjlvtqnbhcprsg: first marker after character 10
zcfzfwzzqfrljwzlrfnpqdbhtmscgvjw: first marker after character 11
How many characters need to be processed before the first start-of-packet marker is detected?
"""

def generate_args(file, nchars, size = 1000):
    """
    Generate arguments for find_n_chars -subprocesses
    Takes a chunk of file and returns the chunk, the starting index and the number of characters to find
    """
    chars = 0
    with open(file,"r") as f:
        print(f"Read file '{file}'")
        while True:
            string = f.read(size)
            if not string:
                return
            yield (chars, string, nchars)
            chars += size

def find_n_chars(start : int, text : str, n : int):
    """ Find a spot, where there are n unique characters in a row

    Args:
        start (int): What is the index of start of the text
        text (str): The text to go through
        n (int): How many unique characters in a row, mark the start of a relevant packet

    Returns:
        Tuple[bool, int]: (True, index) if found, (False, index) if not found
    """
    if not text:
        return (None, None)
    chars = []
    for nchar, c in enumerate(text):
        if c in chars:
            char_ind = chars.index(c)
            if char_ind + 1 == len(chars)-1:
                chars = [chars[-1],c]
            else:
                chars = chars[char_ind+1:] + [c]
        else:
            chars.append(c)
        if len(chars) == n:
            print(chars)
            return True, start + nchar+1
    return False, start + nchar+1

def find_n_chars_wrap(*args):
    """ Wrapper for find_n_chars, so it can be used in multiprocessing.Pool"""
    args = args[0]
    return find_n_chars(args[0],args[1],args[2])


if __name__ == "__main__":
    # The bigboy.txt is a 10 Million character long file
    # NOTE: This is fast, but it isn't guaranteed to work!!!
    # To fix it, the splits should overlap
    start_time = time.time()
    with multiprocessing.Pool(10) as pool:
        file = "./bigboy.txt"
        n = 4
        size = 10000000//10
        str_gen = generate_args(file, n,size)
        gen = pool.imap(find_n_chars_wrap,str_gen,chunksize=1)
        while gen:
            try:
                res = next(gen)
            except StopIteration as si:
                break
            if res[0]:
                print(res)
                break
    print(f"Found in {round(time.time() - start_time,2)}")