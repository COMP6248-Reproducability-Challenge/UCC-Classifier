import numpy as np
import matplotlib.pyplot as plt
import matplotlib

ae_train_loss, ucc_train_loss, total_train_loss = np.loadtxt("loss_data/2020_04_05__20_31_24_128000/losses_2000.csv", skiprows=1, dtype=float, delimiter=",", unpack=True)


font = {'family': 'normal',
        'weight': 'bold',
        'size': 18}

matplotlib.rc('font', **font)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

plt.plot(ae_train_loss, label="Autoencoder Loss")
plt.plot(ucc_train_loss, label="Classifier Loss")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.tight_layout()
plt.savefig("loss_data/2020_04_05__20_31_24_128000/loss_plot.pdf")
