from __future__ import division
from subprocess import call
from os import listdir
from os import path
from datetime import datetime
import numpy
import random
import backports.lzma as lzma
import multiprocessing
import pickle
import re
from multiprocessing import Pool

NUMBER_OF_ANCHORS = 100
COMPARISONS_PER_FILE = 40
BENCHMARKS_FOR_INITIAL_AVERAGE = 30
DEVIATION_FACTOR = 0.5
DATA_DIR = "data"
TARGET_DIR = "fragmented_log"
ANCHOR_DIR = "representative_anchors"

lzma_filters = my_filters = [
    {
      "id": lzma.FILTER_LZMA2,
      "preset": 9 | lzma.PRESET_EXTREME,
      "dict_size": 500000000,  # 500 MB, should be large enough.
      "lc": 2, # literal context (lc = 4 actually makes distance matrix worse!, lc = 2 or lc = 3 doesn't matter at all.)
      "lp": 0,
      "pb": 0, #default = 2, set to 0 if you assume ascii
      "mode": lzma.MODE_NORMAL,
      "nice_len": 273, #max
      "mf": lzma.MF_BT4
    }
]

compressed_sizes = {}

def ncd(fa, fb):
    Za = compressed_sizes[fa]
    Zb = compressed_sizes[fb]
    Zab = Z(contents[fa] + contents[fb])
    return (Zab - min(Za, Zb)) / max(Za, Zb)

def Z(contents) :
  return len(lzma.compress(contents, format=lzma.FORMAT_RAW, filters= lzma_filters))

def compress_file(file_name):
   return [file_name, Z( contents[file_name] )]

def calculate_compressed_sizes():
    print("Compressing all files.")
    p = Pool()
    sizes = p.map(compress_file, paths)
    compressed_sizes.update(sizes)
    print("Done compressing all files.")

def get_random_anchor_and_benchmarks():
    random_files = random.sample(paths, COMPARISONS_PER_FILE + 1)
    return [random_files[0], random_files[1:]]

def calculate_initial_average_and_std():
    averages = []
    for _ in range(BENCHMARKS_FOR_INITIAL_AVERAGE):
        anchor, benchmarks = get_random_anchor_and_benchmarks()
        averages.append( average_ncd(anchor, benchmarks) )

    return [numpy.mean(averages), numpy.std(averages)]

def average_ncd(anchor, benchmarks):
    return numpy.mean( [ncd(anchor, b) for b in benchmarks] )

def from_same_file(fragment, list_of_fragments):
    parent_file = fragment.split("_")[0]
    for f in list_of_fragments:
        other_parent_file = f.split("_")[0]
        if parent_file == other_parent_file:
            return True

    return False

# make initial average: for 10 files, calculate the ncd with 100 other files.
# now, select files that score above average (averaged over 100 / 1000 files)
# keep the files that are above average close to other files.
# Adjust initial average somehow, repeat process.

def select_representative_anchors(number_of_anchors, target_dir):
    global contents
    global paths

    contents = { pathname : open( path.join(target_dir, pathname) ).read() for pathname in listdir(target_dir) }
    paths = contents.keys()

    calculate_compressed_sizes()
    average_distance, standard_deviation = calculate_initial_average_and_std()

    print "I computed an initial average of " + str(average_distance) + ".\n"
    print "I computed a standard deviation of " + str(standard_deviation) + "."

    anchors = []
    tries = 0

    while len(anchors) < number_of_anchors:
        tries += 1

        possible_anchor, benchmarks = get_random_anchor_and_benchmarks()
        if possible_anchor in anchors:
            print "Whoops, already got that one."
        # if from_same_file(possible_anchor, anchors):
        #     print "Whoops, that fragment is from a file we've already got."
        #     continue

        distance = average_ncd(possible_anchor, benchmarks)
        print "That one had " + str(distance) + "."

        if distance < average_distance - DEVIATION_FACTOR * standard_deviation:
            print "(***) Found one with " + str(distance) + "."
            anchors.append(possible_anchor)

    print "Tried " + str(tries) + " files."
    return anchors


def main():
    for target_dir in ["blogs_female_cleaned"]:#["fragmented_csv", "fragmented_jpg", "fragmented_html"]:

        print "starting with " + target_dir + "."

        anchors = select_representative_anchors(NUMBER_OF_ANCHORS, path.join(DATA_DIR, target_dir))
        print anchors
        file_name = target_dir + "_" + str(NUMBER_OF_ANCHORS) + "anchors_" + str(datetime.now()) + ".pickle"

        with open(path.join(ANCHOR_DIR, file_name), "w") as f:
            pickle.dump(anchors, f)

if __name__ == "__main__":
    main()
