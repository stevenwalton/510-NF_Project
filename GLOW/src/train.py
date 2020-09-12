import numpy as np
import torch
from torch import nn
from torch import optim
from torch.distributions import Normal
import torchvision
from torchvision import transforms

from model import GLOW
import utils

def train(dataset_name = 'celeb',
          nepochs=1,
          eval_freq=10,
          lr=1e-4,
          batch_size=20,
          cuda=False,
          channels=3,
          size=32,
          #depth=24,
          depth=1,
          #n_levels=4,
          n_levels=3,
          std=0.7,
          n_bits=4,
          download=True,
          num_workers=8,
          ):
    if "celeb" in dataset_name:
        print(f"Using CelebA Data Set")
        transform = transforms.Compose([
            transforms.CenterCrop(192),
            transforms.Resize(64),
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
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    gaussian = Normal(torch.zeros(1).to(device), torch.ones(1).to(device))

    prior = gaussian
    model.train()
    epoch_loss_array = torch.zeros(nepochs)
    for epoch in range(nepochs):
        running_loss = 0
        for i, (imgs, _) in enumerate(trainloader):
            optimizer.zero_grad()
            zs, log_det = model(imgs.to(device))
            # Fix to torch.sum
            prior_logprob = sum(prior.log_prob(z).sum([1,2,3]) for z in zs)
            log_prob = prior_logprob + log_det

            # Bits per pixel
            log_prob /= (np.log(2) * channels * size * size)
            loss = -log_prob.mean()
            if loss == float('inf'):
                print(f"Something went wrong! Loss is {loss}")
                exit(1)
            running_loss -= loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 50) # Why this
            optimizer.step()
        epoch_loss_array[epoch] = loss.item()
        print(f"Epoch: {epoch} finished with loss {loss}")

        if epoch % eval_freq == 0 or epoch == (nepochs-1):
            with torch.zero_grad():
                samp = sample(prior,
                              n_samples=n_samples,
                              n_levels=n_levels,
                              init_channel=channels,
                              init_hw=size,
                              std=std)
                xs, log_prob = model.backward(samp)
                sample_imgs = torchvision.utils.make_grid(xs)
                fig, ax = plt.subplots()
                ax.imshow(sample_imgs)
                ax.set_title("GLOW: Log Prob {log_prob:.4f}")
                plt.savefig(f"GLOW_sample_epoch_{epoch}.png")
                
