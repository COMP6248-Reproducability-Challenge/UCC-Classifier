# AUTHORS: Sai Pandian, Mohammed Mosuily, Sameen Islam 
# TODO: Currently broken (probably due to custom layers in model)

import numpy as np
import pydot
from model import Keras_Model
from tensorflow.keras.utils import plot_model

# the min/max number of possible classes
min_classes = 1
max_classes = 4

# get initial approx num classes and num samples per class
num_classes =  max_classes - min_classes + 1
num_samples_per_class = int(20 / num_classes)

# number of instances
num_instances = 32

# subsets in MNIST data
subsets_elements_arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# get batch size
batch_size = num_classes * num_samples_per_class

model = Keras_Model(patch_size=28, num_instances=32, num_classes=num_classes, learning_rate=1e-4, num_bins=11, num_features=10, batch_size=batch_size)

plot_model(model, rankdir="LR", show_shapes=False, show_layer_names=False)
