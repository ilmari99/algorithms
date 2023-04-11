import multiprocessing
import os
import time

def generate_args(file, nchars, size = 1000):
    chars = 0
    with open(file,"r") as f:
        print(f"Read file '{file}'")
        while True:
            string = f.read(size)
            if not string:
                return
            yield (chars, string, nchars)
            #print(f"Distributed {chars} characters")
            chars += size

def find_n_chars(start : int, text : str, n : int):
    #print("Start", start)
    #print("Text: ", text)
    if not text:
        return (None, None)
    chars = []
    for nchar, c in enumerate(text):
        #print(chars)
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
    args = args[0]
    return find_n_chars(args[0],args[1],args[2])


if __name__ == "__main__":
    #gen = generate_chars("input.txt",2000)
    #res = find_4_chars(*next(gen))
    #res = find_4_chars(0,"nppdvjthqldpwncqszvftbrmjlhg")
    #print(res)
    #exit()
    start_time = time.time()
    with multiprocessing.Pool(os.cpu_count()) as pool:
        file = "C:\\Users\\ilmari\\Desktop\\Python\\moska\\bigboy.txt"
        n = 14
        size = 10000000//(os.cpu_count() + 1)
        str_gen = generate_args(file, n,size)
        gen = pool.imap(find_n_chars_wrap,str_gen,chunksize=1)
        while gen:
            try:
                res = next(gen)
                #print(f"Results {res}")
            except StopIteration as si:
                break
            if res[0]:
                print(res)
                break
    print(f"Found in {round(time.time() - start_time,2)}")