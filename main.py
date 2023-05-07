from parameters.hyperparameters import HyperParameters
from train_model import train, train_valid, get_model_filename
from test_model import test
from utils import parse_args


def main(hyperparameters):
    model = train(hyperparameters)
    test(model)


def main_new(hyperparameters):
    model = train_valid(hyperparameters)
    test(model)


def get_hyperparameters():
    args = parse_args()

    train_path = './data/plant-seedlings-classification-cs429529/train'
    save_every_num_epoch = 1

    # get and print parameters
    # hyperparameters = HyperParameters(args, TRAIN_PATH)

    args_get = False
    if args_get:
        # from args
        input_args = [args, train_path, save_every_num_epoch]
        index_args = list(range(len(input_args)))
    else:
        # from custom values
        # pretrained_model_custom = f"model-0.96-best_train_acc.pth"
        # pretrained_model_custom = f"state_dict-batch_size=32-lr=0.0001-0.59.pth"
        pretrained_model_custom = None
        # index_args = list(range(6))
        input_args = [1, 32, 0.0001, 4, train_path, pretrained_model_custom, save_every_num_epoch]
        index_args = list(range(len(input_args)))

    assert len(index_args) == len(input_args)
    dict_args = {}
    for index, input in zip(index_args, input_args):
        dict_args[f"{index}"] = input

    hyperparameters = HyperParameters(args_get=args_get, **dict_args)

    # acc = 0
    # hyperparameters.pretrained_model_path = get_model_filename(acc)

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
    # main(hyperparameters)
    main_new(hyperparameters)
