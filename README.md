# Classification using the normalized compression distance

```python
pip install -r requirements.txt
```

to install dependencies.

In `classify.py`, specify the classes you want to train the classifier on. For example:

```python
BASE_DIR = "data"
DIRECTORIES = [
  "fragmented_csv",
  "fragmented_jpg"
]
```

The files for each class should be in their own directory. The class label is the file extension.

Select the number of anchors per class and the number of items per class:

```python
ITEMS_PER_CLASS = 1000
ANCHORS_PER_CLASS = 10
```

Then run `python classify.py`. A report is automatically written to
`reports/` and to stdout. For parameters mentioned above:

```
precision    recall  f1-score   support

.csv     0.9950    0.9950    0.9950      1000
.jpg     0.9950    0.9950    0.9950      1000

avg / total     0.9950    0.9950    0.9950      2000

```


`CV` specifies the number of folds used in cross validation.
`GRID_SEARCH_CV` specifies the number of folds used in cross validation
for estimating the fitness of the model parameters. For each training
partition of the data set, the best parameters within the specified space
are found, and are used to predict the test partition.

You can specify the parameter space as follows:

```python

PARAM_GRID = [
    {'kernel': ['rbf'], 'gamma': [ 2 ** n for n in numpy.arange(-9, 2, 1)
    ], 'C': [ 2 ** n for n in numpy.arange(-2, 9, 1) ] }
] 

``` 

### References 

[scikit-learn](http://scikit-learn.org/stable/modules/svm.html) 

[A Practical Guide to Support Vector
Classification](http://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf)





