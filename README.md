# Weakly Supervised Clustering by Exploiting Unique Class Count

Reproduced from ICLR 2020 paper titled "Weakly Supervised Clustering By
Exploiting Unique Class Count"

ICLR Submission: https://openreview.net/forum?id=B1xIj3VYvr

This technique is a weakly supervised learning based clustering framework
performs comparable to that of fully supervised learning models by exploiting
unique class count.

![ucc classifier](https://github.com/samisnotinsane/iclr-breast-cancer-classification/blob/main/ucc-classifier.png?raw=true)


This paper has proposed a new type of weakly supervised clustering / multiple
instance learning (MIL) algorithm in which bags of instances (data points) are
labeled with a "unique class count (UCC)" bag-level label, rather than any
instance-level labels.

The algorithm is evaluated on MNIST dataset.

To train the model, first create the virtual environment using conda and activate it.
```sh
conda env create -f environment.yml
conda activate deep_learning
```

Now you can train the model contained in the MNIST\_Model folder using:
```sh
python train_model.py
```

The model can be evaluated using the original paper authors' code contained in
the MNIST\_Model/Original\_Paper\_Code folder.
