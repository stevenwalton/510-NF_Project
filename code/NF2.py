import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.distributions import Uniform, TransformedDistribution, MultivariateNormal
from torch.distributions.transforms import SigmoidTransform, AffineTransform
from torchvision import transforms
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
numworkers = 10
batch_size = 128
transform = transforms.Compose([
    transforms.ToTensor(),
    ])
mnist_data = datasets.MNIST('./datasets', 
                            download=True,
                            transform=transform,
                            )
data = DataLoader(mnist_data,
                  batch_size=batch_size,
                  shuffle=True,
                  num_workers=numworkers,
                  )

def LinearBlock(_in,_out,reg=True, inplace=True):
    layers = []
    layers.append(nn.Linear(_in,_out))
    if reg:
        layers.append(nn.ReLU(inplace=inplace))
    return layers

class Coupling(nn.Module):
    def __init__(self,dim, numLayers=5):
        super(Coupling,self).__init__()
        layers = []
        for i in range(numLayers):
            if i == 0:
                layers.append(nn.Linear(dim//2,1000))
            elif i == numLayers-1:
                layers.append(nn.Linear(1000,dim//2))
            else:
                layers.append(nn.Linear(1000,1000))
            layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)

    def forward(self,x):
        print("Coupling")
        x = self.layers(x)
        return x


class Additive(nn.Module):
    def __init__(self,dim, reverse=False, numLayers=5):
        super(Additive,self).__init__()
        self.numLayers = numLayers

        self.layers = Coupling(dim)
        self.reverse = reverse

    def split(self,x):
        if len(x.size()) == 4:
            B,C,H,W = x.size()
            size = C*H*W
        else:
            B,size = x.size()
        x = x.reshape(B,-1)
        x1 = x[:,size//2:]
        x1.requires_grad_(True)
        x2 = x[:,:size//2]
        x2.requires_grad_(True)
        return x1,x2

    def forward(self,x,y):
        print("Into Additive")
        if not self.reverse:
            x1,x2 = self.split(x)
        else:
            x2,x1 = self.split(x)
        y1 = x1
        y2 = x2 + self.layers(x1)
        return torch.cat((y1,y2),dim=1)

    def backward(self,x):
        if not self.reverse:
            y1,y2 = self.split(x)
        else:
            y2,y1 = self.split(x)
        x1 = y1
        x2 = y2 - self.layers(y1)
        return torch.cat((x1,x2),dim=1)

class Scale(nn.Module):
    def __init__(self,dim=28**2):
        super(Scale,self).__init__()
        self.scale = nn.Parameter(torch.ones(1,dim,requires_grad=True))

    def forward(self,x):
        log_det = torch.sum(self.scale)
        x = x*torch.exp(self.scale)
        return x,log_det

    def backward(self,x):
        log_det = torch.sum(self.scale)
        x = x*torch.exp(-self.scale)
        return x,log_det


class Nice(nn.Module):
    def __init__(self,dim,numLayers=5):
        super(Nice,self).__init__()

        layers = []
        for i in range(numLayers):
            layers.append(Additive(dim=dim,reverse=(i%2)))
        layers.append(Scale(dim))
        self.layers = nn.ModuleList(layers)

    def forward(self,x):
        print("Into fwd")
        x,log_det = self.layers(x)
        #x = self.layers(x)
        print("got through fwd")
        return x,log_det

    def backward(self,x):
        x,log_prob = self.layers[-1](x)
        return x,log_prob


net = Nice(28**2).to(device)
#print(net)
prior = torch.distributions.Normal(torch.tensor(0.).to(device),
                                   torch.tensor(1.).to(device))
lr = 1e-4
optimizer = optim.Adam(net.parameters(), lr, eps=1e-4)
epochs = 1
losses = np.zeros(epochs)
for epoch in range(epochs):
    running_loss = 0
    for i,(img,label) in enumerate(data):
        optimizer.zero_grad()
        img = img.to(device)
        pred,liklihood = net(img)
        #pred = net(img)
        #log_prob = prior.log_prob(pred).sum(1)
        #loss = -torch.mean(liklihood + log_prob)
        #running_loss += loss.item()
        #loss.backward()
        #optimizer.step()

    epoch_loss = running_loss/(i+1)
    losses[epoch] = epoch_loss
    print(f"Epoch {epoch} completed with loss {epoch_loss}")

net.eval()
h = prior.sample((9,))
pred,liklihood = net.backward(h)
pred = pred.detach().cpu().view(-1,28,28)
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(pred[i],cmap=plt.cm.binary)
plt.subplots_adjust(bottom=0.1, right=0.8,top=0.9)
plt.savefig("NFOut.png")
print(f"Liklihood {liklihood}")
