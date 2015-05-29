from __future__ import division
import matplotlib.pyplot as plt
import backports.lzma as lzma
import os
import io
import pprint
import numpy
import pdb
import pickle
import random
from sklearn import svm
from sklearn import cross_validation
from sklearn import grid_search
from sklearn import metrics


BASE_DIR = "data"
DIRECTORIES = [
    "fragmented_log",
    "fragmented_doc",
    "fragmented_html",
    "fragmented_csv",
    "fragmented_pdf",
    "fragmented_txt",
    "fragmented_xls",
    "fragmented_xml",
    "fragmented_ps",
    "fragmented_ppt",
    "fragmented_eps"
]

ANCHORS_PER_TYPE = 10
TEST_SAMPLES_PER_TYPE = 100
TRAINING_SAMPLES_PER_TYPE = 400

contents = {}
compressed_sizes = {}


def prepare_data():
    anchors, training_samples, test_samples = [], [], []

    for directory in DIRECTORIES:
        anchor_set, training_set, test_set = partition(directory)
        contents.update( { file_name : io.FileIO( os.path.join(BASE_DIR, directory, file_name) ).readall() for file_name in anchor_set + training_set + test_set} )
        anchors += anchor_set
        training_samples += training_set
        test_samples += test_set

    compressed_sizes.update( { file_name : Z(contents[file_name]) for file_name in contents } )
     
    training_items = extract_data_items(training_samples, anchors)
    test_items     = extract_data_items(test_samples, anchors)

    return [training_items, test_items] 


def partition(directory):
    # Maybe randomize? Then anchors etc. are different each time.
    # Less reproducability, but less bias
    file_names = os.listdir( os.path.join(BASE_DIR, directory) )
    random.shuffle(file_names)

    anchors = file_names[:ANCHORS_PER_TYPE]
    training_samples = file_names[ANCHORS_PER_TYPE:ANCHORS_PER_TYPE + TRAINING_SAMPLES_PER_TYPE]
    test_samples = file_names[-TEST_SAMPLES_PER_TYPE:]

    return [anchors, training_samples, test_samples]


def extract_data_items(training_samples, anchors):
    data_items = []

    for sample in training_samples:
        feature_vector = extract_features(sample, anchors)
        label = os.path.splitext(sample)[1]
        data_items.append([feature_vector, label])
        if random.random() < 0.05:
            print("I'm busy with a " + label + " file.")

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


def generate_classifier(training_items):
    tuned_parameters = [
        {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}
    ]

    classifier = svm.SVC(kernel="rbf", C=32, gamma=8)

    coordinates = numpy.asarray( [ item[0] for item in training_items ] )
    labels      = numpy.asarray( [ item[1] for item in training_items ] )

    classifier.fit(coordinates, labels)

    return classifier


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

def serialize(training_items, test_items):
    training_file = os.path.join("feature_vectors", "-".join(["training", str(TRAINING_SAMPLES_PER_TYPE), "samples", str(ANCHORS_PER_TYPE), "anchors"] + DIRECTORIES) + ".pickle")
    testing_file  = os.path.join("feature_vectors", "-".join(["testing", str(TEST_SAMPLES_PER_TYPE), "samples", str(ANCHORS_PER_TYPE), "anchors"] + DIRECTORIES) + ".pickle")

    with open(training_file, "wb" ) as f:
        pickle.dump(training_items, f)

    with open(testing_file, "wb" ) as f:
        pickle.dump(test_items, f)


def deserialize():
    training_file = os.path.join("feature_vectors", "-".join(["training", str(TRAINING_SAMPLES_PER_TYPE), "samples", str(ANCHORS_PER_TYPE), "anchors"] + DIRECTORIES) + ".pickle")
    testing_file  = os.path.join("feature_vectors", "-".join(["testing", str(TEST_SAMPLES_PER_TYPE), "samples", str(ANCHORS_PER_TYPE), "anchors"] + DIRECTORIES) + ".pickle")

    if os.path.isfile(training_file) and os.path.isfile(testing_file): 
        with open(training_file, "rb" ) as f:
            training_items = pickle.load(f)

        with open(testing_file, "rb" ) as f:
            test_items = pickle.load(f)

        return [training_items, test_items]
    else:
        return None

def main():
    deserialized_vectors = deserialize()
    if deserialized_vectors:
        training_items, test_items = deserialized_vectors
    else:
        print("Couldn't find serialized data. Gonna prepare it by calculating ncd's. Could take a while...")
        training_items, test_items = prepare_data()
        serialize(training_items, test_items)

    classifier = generate_classifier(training_items)

    mistakes = {}

    for test_item in test_items:
        feature_vector = test_item[0]
        correct_label  = test_item[1]

        prediction = classifier.predict(feature_vector)[0]
        print("my prediction is " + prediction + " and the correct answer is " + correct_label + ".")
        
        if prediction != correct_label:
            mistakes[correct_label] = mistakes.get(correct_label, 0) + 1

    pretty_print(mistakes) 
    print("I made " + str(sum(mistakes.values())) + " mistakes on " + str(len(test_items)) + " test items.")

main()


