import matplotlib.pyplot as plt

x = [1e1, 1e2, 1e3, 1e4]
train_y = [-80.54472148949841, -74.99444935893663, -67.59317395746378, -60.61228444225903]
val_y = [-87.97153090660595, -80.83229271827628, -70.45566558750289, -61.08562092334479]

plt.xscale('log')
plt.plot(x, train_y, 'ro-', label='Train')
plt.plot(x, val_y, 'bo-', label='Validation')

plt.xlabel('#Training Sequences')
plt.ylabel('Avg. Log-Likelihood')

plt.xticks(x)

plt.legend()
plt.show()