import numpy as np
import torch
from torch import nn
from torch import optim
from torch.distributions import Normal
import torchvision
from torchvision import transforms
from matplotlib import pyplot as plt

from model import GLOW
import utils

def train(dataset_name = 'celeb',
          nepochs=1,
          eval_freq=10,
          lr=0.001,
          batch_size=1,
          cuda=False,
          channels=3,
          size=256,
          depth=32,
          n_levels=6,
          n_bits=5,
          download=True,
          num_workers=8,
          n_samples=9,
          #stds = [0.,0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
          stds=[0.75],
          ):
    batch_size=1
    #size=224
    assert(n_levels > 1),f"Must have more than 1 level, even for testing"
    if "celeb" in dataset_name:
        print(f"Using CelebA Data Set")
        transform = transforms.Compose([
            transforms.CenterCrop(size),
            #transforms.Resize(64),
            #transforms.Resize(size),
            transforms.Lambda(lambda im: np.array(im, dtype=np.float32)),
            transforms.Lambda(lambda x: np.floor(x / 2**(8 - n_bits)) / 2**n_bits), #bit reduction
            transforms.ToTensor(),
            transforms.Lambda(lambda t: t + torch.rand_like(t) / 2**n_bits) # dequantization like we did on NICE
        ])  

        trainset = torchvision.datasets.CIFAR10(root='./cifar', 
                                                train=True,
                                                download=download, 
                                                transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, 
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=num_workers)

        classes = ('airplane', 
                   'automobile', 
                   'bird', 
                   'cat', 
                   'deer', 
                   'dog', 
                   'frog', 
                   'horse', 
                   'ship', 
                   'truck')
    elif "cifar" in dataset_name:
        print(f"Using CIFAR10 Data Set")
        exit(1)
        transform = transforms.Compose([transforms.CenterCrop(192),  #
                                               transforms.Resize(64),
             transforms.Lambda(lambda im: np.array(im, dtype=np.float32)),
             transforms.Lambda(lambda x: np.floor(x / 2**(8 - n_bits)) / 2**n_bits),
             transforms.ToTensor(),
             transforms.Lambda(lambda t: t + torch.rand_like(t) / 2**n_bits)
            ])  

        trainset = torchvision.datasets.CelebA(root='./celeba', 
                                               split='train', 
                                               transform=transform, 
                                               download=download)
        trainloader = torch.utils.data.DataLoader(trainset, 
                                                  batch_size=batch_size, 
                                                  shuffle=True, 
                                                  pin_memory=True,
                                                  num_workers=num_workers)
    else:
        print(f"Don't know dataset {dataset_name}")
        exit(1)
    if cuda is True and torch.cuda.device_count() > 1:
        device = torch.device('cuda')
        model = nn.DataParallel(GLOW(channels=channels,
                                     depth=depth,
                                     n_levels=n_levels)).cuda()
    elif cuda is True:
        device = torch.device('cuda')
        model = GLOW(channels=channels,
                     depth=depth,
                     n_levels=n_levels).cuda()
    else:
        device = torch.device('cpu')
        model = GLOW(channels=channels,
                     depth=depth,
                     n_levels=n_levels)
    # Paper uses SGD
    optimizer = optim.SGD(model.parameters(), lr=lr)
    #optimizer = optim.Adam(model.parameters(), lr=lr)
    gaussian = Normal(torch.zeros(1).to(device), torch.ones(1).to(device))

    prior = gaussian
    model.train()
    epoch_loss_array = torch.zeros(nepochs)
    # AMP
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(nepochs):
        running_loss = 0
        for i, (imgs, _) in enumerate(trainloader):
            if i % 100 is 0:
                print(f"Batch {i}/{len(trainset)/batch_size}")
                torch.cuda.empty_cache()
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                zs, log_det = model(imgs.to(device))
            # Fix to torch.sum
            prior_logprob = sum(prior.log_prob(z).sum([1,2,3]) for z in zs)
            log_prob = prior_logprob + log_det

            # Bits per pixel
            log_prob /= (np.log(2) * channels * size * size)
            loss = -log_prob.mean()
            if loss.item() == float('inf'):
                print(f"Something went wrong! Loss is {loss.item()}")
                exit(1)
            running_loss -= loss.detach().cpu().item()
            #loss.backward()
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 50) # 50th norm
            #optimizer.step()
            scaler.step(optimizer)
            scaler.update()
        epoch_loss_array[epoch] = loss.detach().cpu().item()
        torch.cuda.empty_cache()
        print(f"Epoch: {epoch}/{nepochs} finished with loss {loss}")

        if epoch is not 0 and (epoch % eval_freq == 0 or epoch == (nepochs-1)):
            torch.save(model.state_dict(), f"checkpoint_{epoch}.pth")
            #model.load_state_dict(torch.load("checkpoint.pth"), strict=False)
            with torch.no_grad():
                samp = utils.sample(prior,
                              n_samples=n_samples,
                              n_levels=n_levels,
                              init_channels=channels,
                              init_hw=size,
                              std=stds)
                gen_imgs = utils.make_img(model, prior, n_samples, stds,
                        n_levels, channels, 128)
                samp_imgs = torchvision.utils.make_grid(gen_imgs, n_samples).cpu()
                npimg = samp_imgs.numpy()
                npimg = np.transpose(npimg,(1,2,0))
                plt.figure(figsize = (14,12))
                plt.imshow(npimg, interpolation='nearest')
                plt.savefig(f"Epoch_{epoch}_images.png")
                print(f"Saved Epoch_{epoch}_images.png")
    #model.load_state_dict(torch.load("checkpoint_0.pth"), strict=False)
    torch.save(model.state_dict(), "FinalModel.pth")
    with torch.no_grad():
        samp = utils.sample(prior,
                      n_samples=n_samples,
                      n_levels=n_levels,
                      init_channels=channels,
                      init_hw=size,
                      std=stds)
        gen_imgs = utils.make_img(model, prior, n_samples, stds,
                n_levels, channels, 128)
        samp_imgs = torchvision.utils.make_grid(gen_imgs, n_samples).cpu()
        npimg = samp_imgs.numpy()
        npimg = np.transpose(npimg,(1,2,0))
        plt.figure(figsize = (14,12))
        plt.imshow(npimg, interpolation='nearest')
        plt.savefig("FinalOutput.png")
