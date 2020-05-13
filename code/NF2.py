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
batch_size = 1
#batch_size=1
transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.Normalize((0.5),(0.5)),
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

class LinearBlock(nn.Module):
    def __init__(self,_in,_out,reg=True,inplace=True):
        super(LinearBlock,self).__init__()
        self.block = nn.Linear(_in,_out)
        self.reg = nn.ReLU(inplace=inplace)
        self.regBool = reg

    def forward(self,x):
        x = self.block(x)
        if self.regBool:
            x = self.reg(x)
        return x

class Coupling(nn.Module):
    def __init__(self,dim, numLayers=5):
        super(Coupling,self).__init__()
        layers = nn.ModuleList()
        layers.append( LinearBlock(dim//2,1000))
        for i in range(numLayers):
            layers.append(LinearBlock(1000,1000))
        layers.append(LinearBlock(1000,dim//2))
        self.layers = layers

    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return x

    '''
    def backward(self,x):
        for layer in self.layers[::-1]:
            x = layer(x)
        return x
        '''

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

    def forward(self,x):
        if not self.reverse:
            x1,x2 = self.split(x)
        else:
            x2,x1 = self.split(x)
        y1 = x1
        y2 = x2 + self.layers(x1)
        log_det = torch.zeros(x.size()[0])
        return torch.cat((y1,y2),dim=1),log_det

    def backward(self,x):
        if not self.reverse:
            y1,y2 = self.split(x)
        else:
            y2,y1 = self.split(x)
        x1 = y1
        #x2 = y2 - self.layers.backward(x1)
        x2 = y2 - self.layers(x1)
        log_det = torch.zeros(x.size()[0])
        return torch.cat((x1,x2),dim=1),log_det

class Scale(nn.Module):
    def __init__(self,dim=28**2):
        super(Scale,self).__init__()
        self.scale = nn.Parameter(torch.ones(1,dim,requires_grad=True))

    def forward(self,x):
        x = x*torch.exp(self.scale)
        log_det = torch.sum(self.scale,dim=1)
        return x,log_det

    def backward(self,x):
        x = x*torch.exp(-self.scale)
        log_det = torch.sum(-self.scale,dim=1)
        return x,log_det


class Nice(nn.Module):
    def __init__(self,dim,numLayers=5):
        super(Nice,self).__init__()
        self.a = nn.ModuleList([Additive(dim=dim, reverse=(i%2)) for i in range(4)] )
        self.s = Scale(dim)

    def forward(self,x):
        for layer in self.a:
            x,log_det = layer(x)
        x,log_det = self.s(x)
        return x,log_det

    def backward(self,x):
        x,log_prob = self.s(x)
        for layer in self.a[::-1]:
            x,log_det = layer.backward(x)
        return x,log_prob


dim = 28**2
net = Nice(dim).to(device)
#net = Additive(28**2).to(device)
#print(net)
#prior = torch.distributions.Normal(torch.tensor(0.).to(device),
#                                   torch.tensor(1.).to(device))
base_distribution = Uniform(torch.zeros(dim).to(device),
        torch.ones(dim).to(device))
transforms = [SigmoidTransform().inv, AffineTransform(loc=0, scale=1)]
logistic = TransformedDistribution(base_distribution, transforms)

gaussian = MultivariateNormal(torch.zeros(dim).to(device),
        torch.eye(dim).to(device))
prior = gaussian
#prior = logistic
lr = 1e-4
optimizer = optim.Adam(net.parameters(), lr, eps=1e-4)
epochs = 100
losses = np.zeros(epochs)
for epoch in range(epochs):
    running_loss = 0
    for i,(img,label) in enumerate(data):
        optimizer.zero_grad()
        img = img.to(device)
        pred,liklihood = net(img)
        #pred = net(img)
        #log_prob = prior.log_prob(pred).sum(1)
        log_prob = prior.log_prob(pred)
        loss = -torch.sum(liklihood + log_prob)
        running_loss += loss.item()
        net.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss = running_loss/(i+1)
    losses[epoch] = epoch_loss
    print(f"Epoch {epoch} completed with loss {epoch_loss}")

torch.save(net.state_dict(), "NF2.pt")
net.eval()
h = prior.sample((9,))
#print(h)
pred,liklihood = net.backward(h)
#pred,liklihood = net.forward(h)
pred = pred.detach().cpu().view(-1,28,28)
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(pred[i],cmap=plt.cm.binary)
plt.subplots_adjust(bottom=0.1, right=0.8,top=0.9)
plt.savefig("NFOut.png")
print(f"Liklihood {liklihood.item()}")

'''
plt.clf()
#fig,axes = plt.subplots(1)
img = img[0]
#axes[0] = plt.imshow(img.detach().cpu().squeeze().view(28,28),cmap=plt.cm.binary)
pred,_ = net(img.unsqueeze(0))
pred = pred.detach().cpu()
#axes[1] = plt.imshow(pred.view(28,28),cmap=plt.cm.binary)
plt.imshow(pred.view(28,28),cmap=plt.cm.binary)
#pred = pred.view(-1)
#pred = pred.detach().cpu()
#back,_ = net.backward(pred)
#back = back.detach().cpu().squeeze().view(-1,28,28)
#axes[2] = plt.imshow(back, cmap=plt.cm.binary)
#print(back.size())
#plt.imshow(back, cmap=plt.cm.binary)
#print("Axes[2]")
plt.savefig("inout.png")
#pred,_ = net.forward(h)
#pred = pred.detach().cpu().view(-1,28,28)
#for i in range(9):
#    plt.subplot(3,3,i+1)
#    plt.imshow(pred[i],cmap=plt.cm.binary)
#plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
#plt.savefig("NFForward.png")
'''
