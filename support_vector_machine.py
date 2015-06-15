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
from sklearn import preprocessing
import multiprocessing
from multiprocessing import Pool


BASE_DIR = "data"
#DIRECTORIES = ["fragmented_4096_csv", "fragmented_4096_jpg"]
#DIRECTORIES = ["fragmented_gz", "fragmented_jpg"]
DIRECTORIES = [
	#"male_blogs",
	#"female_blogs"
        #"fragmented_csv",
        "fragmented_jpg",
        #"fragmented_txt",
        "fragmented_log",
        #"fragmented_xml",
        #"fragmented_html",
        #"fragmented_csv"
]


ANCHORS_PER_TYPE = 2
TEST_SAMPLES_PER_TYPE = 100
TRAINING_SAMPLES_PER_TYPE = 100

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

    print("Compressing all files.")
    p = Pool()
    sizes = p.map( compress_file, contents.keys() )
    compressed_sizes.update(sizes)
    #compressed_sizes.update( { file_name : Z(contents[file_name]) for file_name in contents } )
    print("Done compressing all files.")
     
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

def compress_file(file_name):
   return [file_name, Z( contents[file_name] )] 


def extract_data_items(training_samples, anchors):

    p = Pool()
    return p.map( extract_data_item, [[sample, anchors] for sample in training_samples] )

    #for sample in training_samples:
        #feature_vector = extract_features(sample, anchors)
        #label = os.path.splitext(sample)[1]
        #data_items.append([feature_vector, label])


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

def pretty_print(obj):
    printer = pprint.PrettyPrinter()
    printer.pprint(obj)


def generate_classifier(training_items):
    random.shuffle(training_items)

    tuned_parameters = [
        #{'kernel': ['rbf'], 'gamma': [2 ** n for n in numpy.arange(-8, 2, 1)], 'C': [2 ** n for n in numpy.arange(-8, 2, 1)] }
        {'kernel': ['rbf'], 'gamma': [ 2 ** n for n in numpy.arange(-7, 0, 1) ], 'C': [ 2 ** n for n in numpy.arange(0, 6, 1) ] } ,
        #{'kernel': ['linear'], 'C': [ 2 ** n for n in numpy.arange(-10, 10, 1) ]}
        #{'kernel': ['poly'], 'degree': [1, 2, 3, 4, 5], 'coef0': [1, 5, 25, 125], 'C': [1, 10, 100]}
        
    ]

    # What does support vector regression (svr) do or mean?
    # What does SVC mean?
    support_vector_classifier = svm.SVC()
    classifier = grid_search.GridSearchCV(support_vector_classifier, tuned_parameters, cv=2)

    #classifier = svm.SVC(kernel="rbf", gamma=4, C=1)

    coordinates = numpy.asarray( [ item[0] for item in training_items ] )
    labels      = numpy.asarray( [ item[1] for item in training_items ] )

    scaler = preprocessing.StandardScaler().fit(coordinates)
    scaled_coordinates = scaler.transform(coordinates)

    print("Gonna fit my classifier. Wish me luck.")
    #classifier.fit(scaled_coordinates, labels)
    classifier.fit(scaled_coordinates, labels)
    print("Done fitting.")

    print("Best parameters found: ")
    print(classifier.best_estimator_)

    return [classifier, scaler]



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
    deserialized_vectors = False#deserialize()
    if deserialized_vectors:
        training_items, test_items = deserialized_vectors
    else:
        print("Couldn't find serialized data. Gonna prepare it by calculating ncd's. Could take a while...")
        training_items, test_items = prepare_data()
        serialize(training_items, test_items)

    classifier, scaler = generate_classifier(training_items)

    coordinates = numpy.asarray( [ item[0] for item in test_items ] )
    labels      = numpy.asarray( [ item[1] for item in test_items ] )

    scaled_test_items = scaler.transform(coordinates)

    mistakes = {}

    for i in range(len(labels)):
        feature_vector = scaled_test_items[i]
        correct_label  = labels[i]

        prediction = classifier.predict(feature_vector)[0]
        
        if prediction != correct_label:
            mistakes[correct_label] = mistakes.get(correct_label, 0) + 1
            print("I predicted a " + prediction + " file, but it was a " + correct_label + ".")

    number_of_mistakes = sum(mistakes.values())
    pretty_print(mistakes) 
    print("I made " + str(sum(mistakes.values())) + " mistakes on " + str(len(test_items)) + " test items.")
    print("That's a hit percentage of " + str((1 - number_of_mistakes / len(test_items)) * 100))

main()


