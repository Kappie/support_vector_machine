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

Then, select the number of anchors per class and the number of items per class:

```python
ITEMS_PER_CLASS = 1000
ANCHORS_PER_CLASS = 10
```

Then run `python classify.py`. A report is automatically written to `reports/` and a shorter version to stdout.
