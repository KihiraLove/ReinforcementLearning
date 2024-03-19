import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, datasets

train = datasets.MNIST("", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))

test = datasets.MNIST("", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

train_set = DataLoader(train, batch_size=10, shuffle=True)
test_set = DataLoader(test, batch_size=10, shuffle=True)

for data in train_set:
    print(data)
    break
x, y = data[0][0], data[1][0]
print(y)
plt.imshow()