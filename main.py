from train_model import train, train_valid
from test_model import test


def main():
    model = train()
    test(model)


def main1():
    model = train_valid()
    test(model)


if __name__ == '__main__':
    main()
