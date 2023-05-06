from parameters.hyperparameters import HyperParameters
from train_model import train, train_valid
from test_model import test
from utils import parse_args


def main(hyperparameters):
    model = train(hyperparameters)
    test(model)


def main1(hyperparameters):
    model = train_valid(hyperparameters)
    test(model)


def get_hyperparameters():
    args = parse_args()

    TRAIN_PATH = './data/plant-seedlings-classification-cs429529/train'

    # get and print parameters
    # hyperparameters = HyperParameters(args, TRAIN_PATH)

    args_get = False
    if args_get:
        # from args
        index_args = list(range(2))
        input_args = [args, TRAIN_PATH]
    else:
        # from custom values
        pretrained_model_custom = f"model-0.96-best_train_acc.pth"
        index_args = list(range(6))
        input_args = [1, 64, 1, 4, TRAIN_PATH, pretrained_model_custom]

    assert len(index_args) == len(input_args)
    dict_args = {}
    for index, input in zip(index_args, input_args):
        dict_args[f"{index}"] = input

    hyperparameters = HyperParameters(args_get=args_get, **dict_args)
    hyperparameters.print_parameters()

    # NUM_EPOCHS = args.epochs
    # BATCH_SIZE = args.batch_size
    # LR = args.lr
    # NUM_WORKERS = args.num_workers
    # PRETRAINED_MODEL = args.pretrained_model_path

    # NUM_EPOCHS = hyperparameters.num_epochs
    # BATCH_SIZE = hyperparameters.batch_size
    # LR = hyperparameters.lr
    # NUM_WORKERS = hyperparameters.num_workers
    # PRETRAINED_MODEL = hyperparameters.pretrained_model_path

    return hyperparameters


if __name__ == '__main__':
    hyperparameters = get_hyperparameters()
    main(hyperparameters)
    # main1(hyperparameters)
