from __future__ import division
import backports.lzma as lzma
import os
import io
import random
import pprint

lzma_filters = my_filters = [
    {
        "id": lzma.FILTER_LZMA2, 
        "preset": 9 | lzma.PRESET_EXTREME,
        #"dict_size": 5000000,  
        "lc": 3, # literal context (lc = 4 actually makes distance matrix worse!, lc = 2 or lc = 3 doesn't matter at all.)
        #"lp": 0,
        "pb": 0, #default = 2, set to 0 if you assume ascii
        #"mode": lzma.MODE_NORMAL,
        "nice_len": 273,#273, #max
        #"mf": lzma.MF_BT4
    }
]


DIR = "/Users/geertkapteijns/Code/Ruby/clustering/test_files/34-mammals/"
NUMBER_OF_FILES = 2
NUMBER_OF_TRIALS = 1000


def Z(contents):
  return len(lzma.compress(contents, format=lzma.FORMAT_RAW, filters= lzma_filters))


def main():
    ratio = 0.0
    for _ in range(NUMBER_OF_TRIALS):
        files = { f : io.FileIO( os.path.join(DIR, f) ).readall() for f in random.sample(os.listdir(DIR), NUMBER_OF_FILES) }
        
        big_file = "".join(files.values())
        
        ratio += ( Z(big_file) / len(big_file) ) / NUMBER_OF_TRIALS

    print ratio
    print "\n"
    print pprint.pformat(lzma_filters)

main()
