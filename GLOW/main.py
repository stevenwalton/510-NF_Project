import sys
sys.path.append('src')
import train

def main():
    train.train(dataset_name='cifar',
                nepochs=200,
                eval_freq=5,
                lr=0.00001,
                batch_size=200,
                size=32,
                depth=32,
                n_bits=5,
                n_levels=3,
                stds=[0.99],
                cuda=True,
                n_samples=16,
                start_at=0,
                )

if __name__ == '__main__':
    main()
