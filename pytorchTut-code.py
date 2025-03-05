#%%
import torch
import torchvision
import torchvision.transforms as transforms

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 10
num_workers=0

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=num_workers)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=num_workers)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# show image
import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

#%%
# create a net
import torch.nn as nn
import torch.nn.functional as F

# ====== vanilla net ======
# class Net(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = torch.flatten(x, 1) # flatten all dimensions except batch
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# ===== BN =====
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # 第一个卷积层 + BatchNorm
        self.conv1 = nn.Conv2d(3, 32, 3,padding=1)  # 输入通道3，输出通道6，卷积核5x5
        self.bn1 = nn.BatchNorm2d(32)      # BatchNorm，通道数为6
        # 池化层
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, 3,padding=1)  
        self.bn2 = nn.BatchNorm2d(64)     

        self.conv3 = nn.Conv2d(64,128,3,padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128*4*4, 128)  
        self.fc2 = nn.Linear(128, 10) 

        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # 第一层：卷积 -> BatchNorm -> ReLU -> 池化
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # 第二层：卷积 -> BatchNorm -> ReLU -> 池化
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        # 展平操作
        x = torch.flatten(x, 1)  # 展平除批次维度外的所有维度
        # 全连接层1 -> BatchNorm -> ReLU
        x = F.relu(self.fc1(x))
        # 全连接层2 -> BatchNorm -> ReLU
        # x = F.relu(self.fc2(x))
        # dropout
        x = self.dropout(x)
        x = self.fc2(x)
        return x


net = Net()
net.to(device)
print(f"number of parameters: {sum(p.numel() for p in net.parameters() if p.requires_grad) / 1_000_000:.2f}M")
# ===== define loss function and optimizer =====
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
# SGD: lr = 0.001, momentum = 0.9
# adam lr = 3e-4, weight_decay = 1e-6
optimizer = optim.Adam(net.parameters(), lr=3e-4, weight_decay=1e-6)


# ===== train the network =====

for epoch in range(10):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device) , data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000== 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')

# ===== save trained weights =====
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)


#%%
# ===================================================
# evaluate the networks performance on whole data set
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %') 

# SGD: lr = 0.001, momentum = 0.9
# 73 % accuracy with dropout, 75% without dropout, both epoch num = 5
# 76% accuracy without dropout, epch 10






#%%
# ====== test trained network =====
dataiter = iter(testloader)
images, labels = next(dataiter)

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

net = Net()
net.load_state_dict(torch.load(PATH, weights_only=True))

outputs = net(images)
'''
outputs: 
tensor([[-0.7096, -1.4005,  0.1437,  1.6288, -0.2009,  1.3841, -0.3270, -0.2629,
          0.5093, -1.1140],
        [ 3.8252,  7.1294, -2.7016, -3.6907, -4.5535, -4.7634, -4.7776, -5.4013,
          6.3965,  4.8130],
        [ 2.5518,  1.4487, -0.3503, -1.8632, -1.6557, -2.6499, -3.3060, -1.6375,
          4.0620,  1.6534],
        [ 5.5438, -0.4325,  1.2631, -2.5056,  0.5948, -3.7112, -3.8543, -2.0817,
          3.7887, -0.5294]], grad_fn=<AddmmBackward0>)
'''
_, predicted = torch.max(outputs, 1)
# out: (max, max_indices)
'''
variable explain (check outputs above):
_ :tensor([1.6288, 7.1294, 4.0620, 5.5438],grad_fn=<MaxBackward0>)

predicted: tensor([3, 1, 8, 0]) <- indices of the max elem
'''

print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                              for j in range(4)))