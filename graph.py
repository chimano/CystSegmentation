
import matplotlib.pyplot as plt

with open('losses.csv', 'r') as f:
    losses = [float(loss) for loss in f.readline().split(',')]
print(losses)
epochs = list(range(1, len(losses)+1))

plt.plot(epochs,losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.show()