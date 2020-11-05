import numpy as np
import torch
from torch import nn
from torch import optim
from torch.distributions import Normal
import torchvision
from torchvision import transforms
from matplotlib import pyplot as plt
import math

from model import GLOW
import other

def train(dataset_name = 'celeb',
          nepochs=1,
          eval_freq=1,
          lr=0.0001,
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
          start_at=0,
          ):
    #batch_size=10
    #n_levels=4
    #depth=32
    #size=64
    #size=224
    assert(n_levels > 1),f"Must have more than 1 level, even for testing"
    if "celeb" in dataset_name:
        print(f"Using CelebA Data Set")
        transform = transforms.Compose([
            transforms.CenterCrop(192),
            transforms.Resize(size),
            transforms.Lambda(lambda im: np.array(im, dtype=np.float32)),
            transforms.Lambda(lambda x: np.floor(x / 2**(8 - n_bits)) / 2**n_bits), #bit reduction
            transforms.ToTensor(),
            transforms.Lambda(lambda t: t + torch.rand_like(t) / 2**n_bits) # dequantization like we did on NICE
        ])  

        trainset = torchvision.datasets.CelebA(root='./celeba', 
                                               split='train',
                                               download=download, 
                                               transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, 
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=num_workers)
    elif "cifar" in dataset_name:
        transform = transforms.Compose([
            transforms.Lambda(lambda im: np.array(im, dtype=np.float32)),
            transforms.Lambda(lambda x: np.floor(x / 2**(8 - n_bits)) /
                2**n_bits),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: t + torch.rand_like(t) / 2**n_bits)
            ])
        trainset = torchvision.datasets.CIFAR10(root='./cifar',
                                                train=True,
                                                download=download,
                                                transform=transform)
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
    # Paper uses SGD
    optimizer = optim.SGD(model.parameters(), lr=lr)
    #optimizer = optim.Adam(model.parameters(), lr=lr)
    gaussian = Normal(torch.zeros(1).to(device), torch.ones(1).to(device))

    prior = gaussian
    model.train()
    epoch_loss_array = torch.zeros(nepochs)
    # AMP
    #scaler = torch.cuda.amp.GradScaler()
    if start_at != 0:
        model.load_state_dict(torch.load(f"checkpoint_{start_at}.pth"), strict=False)
        print(f"Loading model checkpoint_{start_at}.pth")
    for epoch in range(start_at, nepochs):
        print(f"Running epoch: {epoch}")
        #running_loss = 0
        for i, (imgs, _) in enumerate(trainloader):
            if i % 1000 == 0:
                print(f"Batch {i}/{len(trainset)/batch_size}")
                torch.cuda.empty_cache()
            #with torch.cuda.amp.autocast():
            #    zs, log_det = model(imgs.to(device))
            zs, log_det = model(imgs.to(device))

            # Fix to torch.sum
            prior_logprob = sum(prior.log_prob(z).sum([1,2,3]) for z in zs)
            log_prob = prior_logprob + log_det

            # Bits per pixel
            log_prob /= (np.log(2) * channels * size * size)
            #log_prob /= (math.log(2) * 3 * 32 * 32)
            loss = -log_prob.mean()
            if loss.item() == float('inf'):
                print(f"Something went wrong! Loss is {loss.item()}")
                exit(1)
            # NaNs cause a SVD error and CUDA and Intel MKL crash
            if torch.isnan(loss).any().item():
                print(f"GOT NaNs!!!! ;-;")
                exit(1)
            #running_loss -= loss.detach().cpu().item()
            optimizer.zero_grad()
            loss.backward()
            #scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 50) # 50th norm
            optimizer.step()
            #scaler.step(optimizer)
            #scaler.update()
        #epoch_loss_array[epoch] = loss.detach().cpu().item()
        epoch_loss_array[epoch] = loss.item()
        #torch.cuda.empty_cache()
        print(f"Epoch: {epoch}/{nepochs} finished with loss {loss}")

        if (epoch % eval_freq == 0 or epoch == (nepochs-1)):
            torch.save({'state_dict': model.state_dict(),
                        'epoch': epoch}
                        , f"checkpoint.pth")
            #model.load_state_dict(torch.load("checkpoint.pth")['state_dict'], strict=False)
            with torch.no_grad():
                #samp = other.sample(prior,
                #              n_samples=n_samples,
                #              n_levels=n_levels,
                #              init_channels=channels,
                #              init_hw=size,
                #              std=stds)
                #gen_imgs = other.make_img(model, prior, n_samples, stds,
                #        n_levels, channels, 128)
                #samp_imgs = torchvision.utils.make_grid(gen_imgs, n_samples).cpu()
                #npimg = samp_imgs.numpy()
                #npimg = np.transpose(npimg,(1,2,0))
                #plt.figure(figsize = (14,12))
                #plt.imshow(npimg, interpolation='nearest')
                h = other.sample(prior,
                                 n_samples=n_samples,
                                 n_levels=n_levels,
                                 init_channels=channels,
                                 init_hw=size, 
                                 std=stds)
                xs, _ = model.module.backward(h)
                grid = torchvision.utils.make_grid(xs).cpu()
                npimg = grid.numpy()
                transposed = np.transpose(npimg, (1,2,0))
                plt.figure(figsize=(14,12))
                plt.imshow(transposed, interpolation='nearest')
                plt.savefig(f"Epoch_{epoch}_images.png")
                print(f"Saved Epoch_{epoch}_images.png")
    #model.load_state_dict(torch.load("checkpoint_10.pth"), strict=False)
    torch.save(model.state_dict(), "FinalModel.pth")
    with torch.no_grad():
        samp = other.sample(prior,
                      n_samples=n_samples,
                      n_levels=n_levels,
                      init_channels=channels,
                      init_hw=size,
                      std=stds)
        gen_imgs = other.make_img(model, prior, n_samples, stds,
                n_levels, channels, 128)
        samp_imgs = torchvision.utils.make_grid(gen_imgs, int(np.sqrt(n_samples))).cpu()
        npimg = samp_imgs.numpy()
        npimg = np.transpose(npimg,(1,2,0))
        plt.figure(figsize = (14,12))
        plt.imshow(npimg, interpolation='nearest')
        plt.savefig("FinalOutput.png")
