from __future__ import division
from subprocess import call
from os import listdir
from os import path
import random
import backports.lzma as lzma

DIR = "data/fragmented_csv"

lzma_filters = my_filters = [
    {
      "id": lzma.FILTER_LZMA2, 
      "preset": 9 | lzma.PRESET_EXTREME, 
      "dict_size": 500000000,  # 500 MB, should be large enough.
      "lc": 3, # literal context (lc = 4 actually makes distance matrix worse!, lc = 2 or lc = 3 doesn't matter at all.)
      "lp": 0,
      "pb": 2, #default = 2, set to 0 if you assume ascii
      "mode": lzma.MODE_NORMAL,
      "nice_len": 273, #max
      "mf": lzma.MF_BT4
    }
]

def ncd(fa, fb):
    Za = compressed_sizes[fa]
    Zb = compressed_sizes[fb]
    Zab = Z(contents[fa] + contents[fb])
    return (Zab - min(Za, Zb)) / max(Za, Zb)

def Z(contents):
  return len(lzma.compress(contents, format=lzma.FORMAT_RAW, filters= lzma_filters))

contents = { pathname : open( path.join(DIR, pathname) ).read() for pathname in listdir(DIR) } 


random_path = random.choice( contents.keys() )
random_content = contents[random_path]


hundred_paths = random.sample(contents.keys(), 100)

for p in hundred_paths:
    content = contents[p]
    print ncd(content, random_content)
