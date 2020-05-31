import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.distributions import Uniform, TransformedDistribution, MultivariateNormal
from torch.distributions.transforms import SigmoidTransform, AffineTransform
from torchvision import transforms
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
numworkers = 10
batch_size = 1024
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

def squeeze(x,f=2):
    b,c,h,w = x.size()
    assert(h%f==0 and w%f == 0),f"Squeeze Error: h%f = {h%f} and w%f = {w%f}"
    x = x.reshape(-1,c,h//f,f,w//f,f)
    x = x.permute(0,1,3,5,2,4)
    #x = x.reshape(-1,int(h/f),int(w/f),c*f*f)
    x = x.reshape(-1,c*f*f, int(h/f),int(w/f))
    return x

def unsqueeze(x,f=2):
    b,c,h,w = x.size()
    x = x.reshape(-1,int(c/f/f),f,f,h,w)
    x = x.permute(0,1,4,2,5,3)
    #x = x.reshape(-1,h*f,w*f,int(c/f)**2)
    x = x.reshape(-1,int(c/f/f),h*f,w*f)
    return x

#def split(x):
#    '''
#    Split along channel direction
#    '''
#    b,c,w,h = x.size()
#    x0 = x[:,c//2:,:,:]
#    x1 = x[:,:c//2,:,:]
#    return x0,x1

class Split(nn.Module):
    def __init__(self, c):
        super(Split,self).__init__()
        self.mean = nn.Conv2d(c//2, c//2, kernel_size=3, padding=1)
        self.log_std = nn.Conv2d(c//2, c//2, kernel_size=3, padding=1)

        self.mean.weight.data.zero_()
        self.mean.bias.data.zero_()
        self.log_std.weight.zero_()
        self.log_std.bias.zero_()

    def forward(self,x):
        x0, x1 = x.chunk(2, dim=1)
        mu = self.mean(x0)
        std = self.log_std(x1)
        z = (x1-mu)*torch.exp(-std)
        logdet = -std.sum([1,2,3])
        return x0,x1, logdet

    def backward(self,y0, y1):
        mu = self.mean(y0)
        std = self.log_std(x0)
        x1 = y1 * torch.exp(std) - mu
        legdet = -std.sum([1,2,3])
        return torch.cat((y0,x1),dim=1), logdet


def concat(x0,x1):
    x = torch.cat((x0,x1),dim=1)
    return x

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

class Actnorm(nn.Module):
    def __init__(self,
                 dims,
                ):
        super(Actnorm, self).__init__()
        self.s = nn.Parameter(torch.ones(dims))
        self.b = nn.Parameter(torch.zeros(dims))

    def forward(self,x):
        print(x.size())
        b,c,h,w = x.size()
        s = self.s.view(-1,c,h,w)
        b = self.b.view(-1,c,h,w)
        print(f"actnorm s {s.size()}")
        print(f"actnorm b {b.size()}")
        #y = torch.add(torch.mul(s,x), b)
        y = x*s + b
        #logdet = torch.mul(torch.mul(h,w), torch.sum(torch.abs(s)))
        logdet = h*w*torch.log(self.s.abs()).sum()
        return y, logdet

    def backward(self,y):
        b,c,h,w = y.size()
        x = torch.div(torch.sub(y,self.b),self.s)
        logdet = -h*w*torch.log(self.s.abs()).sum()
        return x

class InvertConv(nn.Module):
    def __init__(self,
                 c,
                ):
        super(InvertConv, self).__init__()
        #W = torch.tensor([[torch.cos(torch.rand(1)), -torch.sin(torch.rand(1))],
        #                   [torch.sin(torch.rand(1)), torch.cos(torch.rand(1))]])
        self.c = c
        W = torch.rand(c,c)
        self.W = nn.Parameter(W)

    def forward(self, x):
        b,c,h,w = x.size()
        #print(f"In invertconv forward x {x.size()}, W {self.W.size()}\
        #        reshaped {(self.W.view(self.c, self.c,1,1)).size()}")
        #y = F.conv2d(x,self.W.view(self.c,self.c,1,1))
        y = F.conv2d(x, self.W.unsqueeze(-1).unsqueeze(-1))
        #logdet = h*w*torch.sum(torch.log(torch.abs(self.s)))
        # Don't understand pivots in torch.lu
        #logdet = h*w*torch.det(torch.abs(self.W))
        logdet = h*w*torch.slogdet(self.W)[1]
        return y, logdet

    def backward(self, y):
        #x = torch.mul(torch.inverse(self.W,y))
        b,c,h,w = y.size()
        Winv = self.W.inverse()
        x = F.conv2d(y, Winv.unsqueeze(-1).unsqueeze(-1))
        logdet = -h*w*torch.slogdet(self.W)[1]
        return x

class FlowStep(nn.Module):
    def __init__(self,
                 c,
                 reverse=False,
                 ):
        super(FlowStep,self).__init__()
        self.an = Actnorm(28**2)
        #self.an = Actnorm(512)
        self.inv = InvertConv(c*4)
        self.affine = AffineCoupling(reverse=reverse)

    def forward(self,x):
        print(f"FlowStep x {x.size()}")
        x,logdet = self.an.forward(x)
        print(f"FlowStep after Actnorm {x.size()}")
        x,logdet = self.inv.forward(x)
        print(f"FlowStep after inv {x.size()}")
        x,logdet = self.affine.forward(x)
        print(f"FlowStep after affine {x.size()}")
        return x, logdet

    def backward(self,y):
        y = self.affine.backward(y)
        y = self.inv.backward(y)
        y = self.an.backward(y)
        return y

class GLOW(nn.Module):
    def __init__(self,
                 c,
                 k=3,
                 L=3
                 ):
        super(GLOW, self).__init__()
        self.k = k
        self.L = L
        self.FS = FlowStep(c)
        self.FSR = FlowStep(c,reverse=True)
        self.splt = Split(4)

    def LStep(self,x):
        print(f"Into LStep {x.size()}")
        x = squeeze(x)
        print(f"After squeeze {x.size()}")
        for i in range(self.k):
            print(f"At step {i} {x.size()}")
            if i%2 == 0:
                x, logdet = self.FS(x)
            else:
                x, logdet = self.FSR(x)
        x,z = self.split(x)
        print(f"After skip {x.size()}")
        return x,z, logdet

    def LStepBack(self,x0,x1):
        x = self.split.backward(x0,x1)
        #x = squeeze(x)
        for i in range(self.k):
            if i%2 == 0:
                x, logdet = self.FS.backward(x)
            else:
                x, logdet = self.FSR.backward(x)
        #x,z = split(x)
        x = self.unsqueeze(x)
        return x, lodget#,z

    def forward(self,x):
        print(f"Glow init {x.size()}")
        for i in range(self.L-1):
            print(f"L step {i}")
            x,z, lodget = self.LStep(x)
        x = squeeze(x)
        #for i in range(self.k):
        #    x,logdet = self.FlowStep(x)
        return x, logdet

    def backward(self,y):
        for i in range(self.k):
            y, lodget = self.FlowStep.backward(y)
        y = unsqueeze(y)
        # There needs to be a concat here
        for i in range(self.L):
            y, lodget = self.LStepBack(y)

        return y, lodget

class Coupling(nn.Module):
    def __init__(self,c):
        super(Coupling, self).__init__()
        self.c = 4#c
        self.NN = nn.Sequential(nn.Conv2d(self.c//2,self.c,kernel_size=3),#, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(self.c,self.c,kernel_size=1),#, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(self.c,self.c,kernel_size=1,padding=1)
                                )
        self.NN[4].weight.data.zero_()
        self.NN[4].bias.data.zero_()

    def forward(self,x):
        #print(f"Inside coupling {x.size()}")
        x = self.NN(x)
        #print(f"After layers {x.size()}")
        logs = x[:,self.c//2:,:,:]
        #print(f"got log s {logs.size()}")
        t = x[:,:self.c//2,:,:]
        #print(f"got t {t.size()}")
        return logs, t

class AffineCoupling(nn.Module):
    def __init__(self,
                 c=2,
                 reverse=False,
                ):
        super(AffineCoupling,self).__init__()
        #self.NN = nn.Sequential(nn.Conv2d(c//2,c,kernel_size=3, padding=1),
        #                        nn.ReLU(),
        #                        nn.Conv2d(c,c,kernel_size=1, padding=1),
        #                        nn.ReLU(),
        #                        nn.Conv2d(c,c,kernel_size=1,padding=1)
        #                        )
        #print(f"NN: {self.NN}")
        #self.NN[4].weight.data.zero_()
        #self.NN[4].bias.data.zero_()
        self.reverse = reverse
        W = torch.zeros(c,c)
        self.W = nn.Parameter(W)
        self.c = c
        self.coup = Coupling(c)

    def split(self,x):
        #if len(x.size()) == 4:
        #    B,C,H,W = x.size()
        #    size = C*H*W
        #else:
        #    B,size = x.size()
        B,C,H,W = x.size()
        size = C*H*W
        x = x.reshape(B,-1)
        x1 = x[:,size//2:]
        x1.requires_grad_(True)
        x2 = x[:,:size//2]
        x2.requires_grad_(True)
        return x1,x2

    def forward(self,x):
        print(f"In Affine {x.size()}")
        if not self.reverse:
            #x_a,x_b = self.split(x)
            x_a,x_b = x.chunk(2,1)
        else:
            #x_b,x_a = self.split(x)
            x_b, x_a = x.chunk(2,1)
        print(f"After split {x_b.size()}")
        #o = self.NN(x_b)
        #print(f"After NN {o.size()}")
        #o = F.conv2d(o, self.W.view(self.c,self.c,1,1))
        #print(f"After conv2d {o.size()}")
        #log_s, t = self.NN.forward(x_b)
        #log_s, t = o.chunk(2,1)
        log_s, t = self.coup(x_b)
        s = torch.exp(log_s)
        #y_a = torch.add(torch.mult(s,x_a),t)
        print(f"x_a {x_a.size()}")
        print(f"s {s.size()}")
        y_a = (x_a*s) + t
        print(f"y_a {y_a.size()}")
        y_b = x_b
        print(f"y_b {y_b.size()}")
        if not self.reverse:
            y = torch.cat((y_a,y_b), dim=1)
        else:
            y = torch.cat((y_b,y_a), dim=1)

        logdet = torch.sum(torch.log(torch.abs(s)))
        return y, logdet

    def backward(self,y):
        if not self.reverse:
            y_a, y_b = self.split(y)
        else:
            y_b, y_a = self.split(y)
        log_s, t = self.NN.backward(y_b)
        s = torch.exp(log_s)
        x_a = torch.div(torch.sub(y_a,t),s)
        x_b = y_b
        if not self.reverse:
            x = torch.cat((x_a, x_b), dim=1)
        else:
            x = torch.cat((x_b, x_a), dim=1)
        return x

dim = 28**2
#net = Nice(dim).to(device)
net = GLOW(c=1).to(device)
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
epochs = 1500
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
    if epoch % 10 == 0:
        h = prior.sample((9,))
        #print(h)
        pred,liklihood = net.backward(h)
        print(f"Epoch {epoch} completed: Loss {epoch_loss}, Liklihood: {liklihood.item()}")
        #pred,liklihood = net.forward(h)
        pred = pred.detach().cpu().view(-1,28,28)
        for i in range(9):
            plt.subplot(3,3,i+1)
            plt.imshow(pred[i],cmap=plt.cm.binary)
        plt.subplots_adjust(bottom=0.1, right=0.8,top=0.9)
        plt.savefig("NFOut_" + str(epoch) + ".png")
        #torch.save(net.state_dict(), "NF2_" + str(epoch) + ".pt")
        plt.clf()

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
