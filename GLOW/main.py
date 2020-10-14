import sys
sys.path.append('src')
import train

def main():
    train.train(nepochs=200,
                cuda=True)

if __name__ == '__main__':
    main()
