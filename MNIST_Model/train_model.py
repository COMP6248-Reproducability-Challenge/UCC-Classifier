import numpy as np
import pandas as pd

from dataset import Dataset
from model_MNIST import UCC_Model_MNIST

# number of epochs:
n_epochs = 1000000

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

# define model
model = UCC_Model_MNIST(image_size=28, num_images=32, num_classes=num_classes, learning_rate=1e-4, num_KDE_bins=11, encoded_size=10, batch_size=batch_size)

# define dataset
split_dataset = Dataset(num_instances=num_instances, num_samples_per_class=num_samples_per_class, digit_arr=subsets_elements_arr, ucc_start=min_classes, ucc_end=max_classes)

# train
ae_train_loss = []
ucc_train_loss = []
total_train_loss = []
ae_test_loss = []
ucc_test_loss = []
total_test_loss = []
for i in range(n_epochs):
    print("Epoch:", str(i+1)+"/"+str(n_epochs))
    print("Getting Next Batch...")
    train_batch = split_dataset.next_batch_train()

    print("Training on Batch...")
    [train_loss_weighted, train_ucc_loss, train_ae_loss, train_ucc_acc, train_ae_acc] = model.classifier_model.train_on_batch(train_batch[0], train_batch[1])
    ae_train_loss.append(train_ae_loss)
    ucc_train_loss.append(train_ucc_loss)
    total_train_loss.append(train_loss_weighted)
    
    # do validation every 100th epoch
    if i % 100 == 0:
        print()
        print("Getting Validation Batch...")
        validation_batch = split_dataset.next_batch_val()
        print("Testing on Batch...")
        [val_loss_weighted, val_ucc_loss, val_ae_loss, val_ucc_acc, val_ae_acc] = model.classifier_model.test_on_batch(validation_batch[0], validation_batch[1])
        ae_test_loss.append(val_ae_loss)
        ucc_test_loss.append(val_ucc_loss)
        total_test_loss.append(val_loss_weighted)

    print()
    print("Subset:                   ", subsets_elements_arr)
    print("Train Loss Weighted:      ", train_loss_weighted)
    print("Train UCC Accuracy:       ", train_ucc_acc)
    print("Train UCC Loss:           ", train_ucc_loss)
    print("Train AE Loss:            ", train_ae_loss)
    print("................................................................")

print("Training Finished!")

print("Saving Model...")
model.classifier_model.save("./Saved_Models/MNIST_classifier_weights_1mil.h5")
model.autoencoder_model.save("./Saved_Models/MNIST_autoenc_weights_1mil.h5")
print("Model Saved!")

print("Saving Loss Values...")
data = {
    "ae_train_loss": ae_train_loss,
    "ucc_train_loss": ucc_train_loss,
    "total_train_loss": total_train_loss,
    "ae_test_loss": ae_test_loss,
    "ucc_test_loss": ucc_test_loss,
    "total_test_loss": total_test_loss
}
df = pd.DataFrame(data)
df.to_csv("losses_1mil.csv", index=False)
print("Saved!")
