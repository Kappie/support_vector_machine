from __future__ import division

import backports.lzma as lzma
import os
import io
import pprint
import numpy
import pdb
import pickle
import random
import multiprocessing
import datetime
from multiprocessing import Pool

from sklearn import svm
from sklearn import cross_validation
from sklearn import grid_search
from sklearn import metrics
from sklearn import preprocessing
from sklearn import pipeline

BASE_DIR = "data"
DIRECTORIES = [
        "fragmented_csv",
        "fragmented_jpg"
]

#DIRECTORIES = [
    #"fragmented_csv",
    #"fragmented_jpg"
#]

ITEMS_PER_CLASS = 3000
ANCHORS_PER_CLASS = 10
GRID_SEARCH_CV = 3
CV = 5

contents = {}
compressed_sizes = {}

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

def prepare_data():
    items, anchors = [], []

    for directory in DIRECTORIES:
        file_names = os.listdir( os.path.join(BASE_DIR, directory) )
        random.shuffle(file_names)

        anchor_set = file_names[:ANCHORS_PER_CLASS]
        item_set = file_names[ANCHORS_PER_CLASS:ANCHORS_PER_CLASS + ITEMS_PER_CLASS]

        contents.update( { file_name : io.FileIO( os.path.join(BASE_DIR, directory, file_name) ).readall() for file_name in anchor_set + item_set} )

        anchors += anchor_set
        items += item_set

    print("Compressing all files.")
    p = Pool()
    sizes = p.map( compress_file, contents.keys() )
    compressed_sizes.update(sizes)
    print("Done compressing all files.")
     
    print("Extracting features with ncd. Might take a while...")
    data_items = extract_features_and_labels(items, anchors)
    print("Done extracting features.")

    feature_vectors = numpy.asarray( [ item[0] for item in data_items ] )
    labels          = numpy.asarray( [ item[1] for item in data_items ] )

    return [feature_vectors, labels]

def compress_file(file_name):
   return [file_name, Z( contents[file_name] )] 


def extract_features_and_labels(items, anchors):
    p = Pool()
    return p.map( extract_data_item, [[item, anchors] for item in items] )

def extract_data_item(args):
    sample, anchors = args
    feature_vector = extract_features(sample, anchors)
    label = os.path.splitext(sample)[1]
    return [feature_vector, label]

def extract_features(sample, anchors):
    return [ ncd(sample, anchor) for anchor in anchors ]

def ncd(fa, fb):
    Za = compressed_sizes[fa]
    Zb = compressed_sizes[fb]
    Zab = Z(contents[fa] + contents[fb])
    return (Zab - min(Za, Zb)) / max(Za, Zb)

def Z(contents):
  return len(lzma.compress(contents, format=lzma.FORMAT_RAW, filters= lzma_filters))

def main():
    vectors, labels = prepare_data()

    param_grid = [
        {'kernel': ['rbf'], 'gamma': [ 2 ** n for n in numpy.arange(-9, 2, 1) ], 'C': [ 2 ** n for n in numpy.arange(-2, 9, 1) ] } ,
    ]

    print("Gonna go out and classify. Wish me luck.")
    support_vector_machine = grid_search.GridSearchCV(svm.SVC(), param_grid, cv = GRID_SEARCH_CV)
    classifier = pipeline.make_pipeline(preprocessing.StandardScaler(), support_vector_machine)

    predicted_labels = cross_validation.cross_val_predict(classifier, vectors, labels, cv = CV)

    file_string = "-".join( DIRECTORIES + [str(ANCHORS_PER_CLASS) + "anchors", str(ITEMS_PER_CLASS) + "items"] ) + ".txt"
    with open( os.path.join("reports", file_string), "w") as f:
        date = "date: " + str(datetime.datetime.now())
        anchors = "anchors per class: " + str(ANCHORS_PER_CLASS)
        grid_search_cv = "grid search cv: " + str(GRID_SEARCH_CV)
        cv = "cv: " + str(CV)
        report = metrics.classification_report(labels, predicted_labels, digits=4)
        print report

        f.writelines("\n".join([date, anchors, grid_search_cv, cv, report]))

main()
