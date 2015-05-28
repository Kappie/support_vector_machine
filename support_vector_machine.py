from __future__ import division
import matplotlib.pyplot as plt
import backports.lzma as lzma
import os
import io
import pprint

from sklearn import datasets
from sklearn import svm
import numpy


DIRECTORIES = ["log", "csv"]
ANCHORS_PER_TYPE = 10
MAX_SAMPLES_PER_TYPE = 50

contents = {}
compressed_sizes = {}


def prepare_data():
    anchors, training_samples = [], []

    for directory in DIRECTORIES:
        anchor_set, training_set = partition(directory)
        contents.update( { file_name : io.FileIO( os.path.join(directory, file_name) ).readall() for file_name in anchor_set + training_set } )
        anchors += anchor_set
        training_samples += training_set

    compressed_sizes.update( { file_name : Z(contents[file_name]) for file_name in contents } )
     
    items = extract_data_items(training_samples, anchors)

    pretty_print(items)

def generate_classifier(data_items):
    classifier = svm.SVC()

    coordinates = [ item[0] for item in data_items ]
    labels      = [ item[1] for item in data_items ]

    classifier.fit(coordinates, labels)

def partition(directory):
    file_names = os.listdir(directory)
    anchors, training_samples = file_names[:ANCHORS_PER_TYPE], file_names[ANCHORS_PER_TYPE:MAX_SAMPLES_PER_TYPE - ANCHORS_PER_TYPE]

    return [anchors, training_samples]


def extract_data_items(training_samples, anchors):
    data_items = []

    for sample in training_samples:
        feature_vector = extract_features(sample, anchors)
        label = os.path.splitext(sample)[1]
        data_items.append([feature_vector, label])

    return data_items 

def extract_features(sample, anchors):
    return [ ncd(sample, anchor) for anchor in anchors ]

def ncd(fa, fb):
    Za = compressed_sizes[fa]
    Zb = compressed_sizes[fb]
    Zab = Z(contents[fa] + contents[fb])
    return (Zab - min(Za, Zb)) / max(Za, Zb)

def pretty_print(obj):
    printer = pprint.PrettyPrinter()
    printer.pprint(obj)


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


def Z(contents):
  return len(lzma.compress(contents, format=lzma.FORMAT_RAW, filters= lzma_filters))


def main():
    data_items = prepare_data()
    classifier = generate_classifier(data_items)





