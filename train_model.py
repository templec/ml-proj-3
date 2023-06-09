import glob
import pickle
import re

import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torchvision.models import resnet18
# from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
import copy
from tqdm import tqdm
import os
import numpy as np

from process_data import PlantsDataset
from utils import parse_args, confusion_matrix
from parameters.hyperparameters import HyperParameters
from constants import update_process_index, get_process_index
from custom_enums import FileOption


# FIXME: gets printed multiple times due to number of workers...4
# INITIAL_PRINT = True
# process_index = get_process_index()
# if INITIAL_PRINT and process_index == 0:
#     args = parse_args()
#
#     TRAIN_PATH = './data/plant-seedlings-classification-cs429529/train'
#
#     # get and print parameters
#     # hyperparameters = HyperParameters(args, TRAIN_PATH)
#
#     args_get = False
#     if args_get:
#         # from args
#         index_args = list(range(2))
#         input_args = [args, TRAIN_PATH]
#     else:
#         # from custom values
#         pretrained_model_custom = f"model-0.96-best_train_acc.pth"
#         index_args = list(range(6))
#         input_args = [1, 64, 0.0005, 4, TRAIN_PATH, pretrained_model_custom]
#
#     assert len(index_args) == len(input_args)
#     dict_args = {}
#     for index, input in zip(index_args, input_args):
#         dict_args[f"{index}"] = input
#
#     hyperparameters = HyperParameters(args_get=args_get, **dict_args)
#     hyperparameters.print_parameters()
#
#     # NUM_EPOCHS = args.epochs
#     # BATCH_SIZE = args.batch_size
#     # LR = args.lr
#     # NUM_WORKERS = args.num_workers
#     # PRETRAINED_MODEL = args.pretrained_model_path
#
#     NUM_EPOCHS = hyperparameters.num_epochs
#     BATCH_SIZE = hyperparameters.batch_size
#     LR = hyperparameters.lr
#     NUM_WORKERS = hyperparameters.num_workers
#     PRETRAINED_MODEL = hyperparameters.pretrained_model_path
#
#     INITIAL_PRINT = False
#     # process_index += 1
#     update_process_index()

NUM_EPOCHS = None
BATCH_SIZE = None
LR = None
NUM_WORKERS = None
PRETRAINED_MODEL = None

TRAIN_PATH = None


if __name__ == "__main__":
    pass


def get_model_filename(best_acc, model_counter):
    return f'model-batch_size={BATCH_SIZE}-lr={LR}-epoch={model_counter}-{best_acc:.2f}.pth'


def get_model_plot_filename(model_counter, prefix):
    return f'{prefix}' \
           f'-batch_size={BATCH_SIZE}-lr={LR}-epoch={model_counter}.png'


def get_file_option_filename(file_option_enum: FileOption):
    return file_option_enum.get_file_option_name() + f"-batch_size={BATCH_SIZE}-lr={LR}.pkl"


def get_file_option_filename_counter():
    return f"model-batch_size={BATCH_SIZE}-lr={LR}-counter.pkl"


def load_file_option_file(file_option_enum: FileOption):
    file_option_enum_filename = get_file_option_filename(file_option_enum)

    with open(file_option_enum_filename, 'rb') as output_file:
        data_output = pickle.load(output_file)

    return data_output


def save_file_option_file(file_option_list: list, file_option_enum: FileOption):
    file_option_enum_filename = get_file_option_filename(file_option_enum)

    with open(file_option_enum_filename, 'wb') as output_file:
        pickle.dump(file_option_list, output_file)


def create_file_option_file(file_option_list: list, file_option_enum: FileOption):
    file_option_enum_filename = get_file_option_filename(file_option_enum)

    if os.path.exists(file_option_enum_filename):
        print(f"LOADING file: {file_option_enum_filename}")
        file_option_list = load_file_option_file(file_option_enum)
    else:
        print(f"CREATING file: {file_option_enum_filename}")
        save_file_option_file(file_option_list, file_option_enum)

    return file_option_list

def load_model_counter():
    model_counter_filename = get_file_option_filename_counter()

    with open(model_counter_filename, 'rb') as output_file:
        data_output = pickle.load(output_file)

    return data_output


def save_model_counter(model_counter):
    model_counter_filename = get_file_option_filename_counter()

    with open(model_counter_filename, 'wb') as output_file:
        pickle.dump(model_counter, output_file)


def create_model_counter(model_counter):
    model_counter_filename = get_file_option_filename_counter()

    if os.path.exists(model_counter_filename):
        print(f"LOADING file: {model_counter_filename}")
        model_counter = load_model_counter()
    else:
        print(f"CREATING file: {model_counter_filename}")
        save_model_counter(model_counter)

    return model_counter


def load_plants_data():
    
    kwargs = {'num_workers': NUM_WORKERS}

    
    data_transform = transforms.Compose(
        [transforms.RandomResizedCrop(224),
         transforms.ToTensor(),
         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
    )

    train_set = PlantsDataset(root_dir=Path(TRAIN_PATH), transform=data_transform)
    data_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE,
                             shuffle=True, **kwargs)

    return train_set, data_loader

def load_plant_validation_data():

    kwargs = {'num_workers': NUM_WORKERS}


    data_transform = transforms.Compose(
        [transforms.RandomResizedCrop(224),
         transforms.ToTensor(),
         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
    )

    train_set = PlantsDataset(root_dir=Path(TRAIN_PATH),
                              transform=data_transform,
                              validation=True,
                              tv_set='t',
                              valid_count=50)
    valid_set = PlantsDataset(root_dir=Path(TRAIN_PATH),
                              transform=data_transform,
                              validation=True,
                              tv_set='v',
                              valid_count=50)

    data_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE,
                             shuffle=True, **kwargs)
    data_loader_valid = DataLoader(dataset=valid_set, batch_size=BATCH_SIZE,
                             shuffle=True, **kwargs)
    
    return train_set, valid_set, data_loader, data_loader_valid


# Load resnet18 pretrained model
def load_model(num_class, hyperparameters):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ResNet
    # model = torch.hub.load('pytorch/vision:v0.5.0',
    #                        'resnet18', pretrained=True)
    # model.fc = nn.Linear(512, num_class)  # by Fan (Not necessary)

    model_counter = hyperparameters.model_counter

    # if args.pretrained_model_path is not None:
    if PRETRAINED_MODEL is not None and model_counter != 0:
        # model.load_state_dict(torch.load(PRETRAINED_MODEL))

        model_filename_regex = fr"model-batch_size={BATCH_SIZE}-lr={LR}-epoch={model_counter}-*"
        model_filenmae_list = glob.glob(model_filename_regex)

        # FIXME: assume only one filename per pattern above (with epoch model number...)
        if model_filenmae_list:
            # FIXME: get path with model counter
            pretrained_model_path = model_filenmae_list[0]
        else:
            raise Exception("NO PATH FOUND...")

        # acc_pattern = \
        #     re.compile(fr"model-batch_size={BATCH_SIZE}-lr={LR}-epoch={model_counter}-(0\.\d+)\.pth")
        # acc_match = re.search(acc_pattern, "./")
        #
        # acc_filepath = acc_match.group(0)
        # acc_num = acc_match.group(1)
        #
        # print(f"RE ACC GROUP 0: {acc_filepath}")
        # print(f"RE ACC GROUP 1: {acc_num}")

        print(f"loading PRETRAINED model...")
        print(f"path: {pretrained_model_path}")
        model = torch.load(pretrained_model_path)
    else:
        # Another method
        print(f"loading DEFAULT model...")
        model = resnet18(pretrained=True)
    
    return model

def train(hyperparameters):
    """
    FIXME: deprecated...

    :param hyperparameters:
    :return:
    """

    global NUM_EPOCHS
    global BATCH_SIZE
    global LR
    global NUM_WORKERS
    global PRETRAINED_MODEL
    global TRAIN_PATH

    NUM_EPOCHS = hyperparameters.num_epochs
    BATCH_SIZE = hyperparameters.batch_size
    LR = hyperparameters.lr
    NUM_WORKERS = hyperparameters.num_workers
    PRETRAINED_MODEL = hyperparameters.pretrained_model_path

    TRAIN_PATH = hyperparameters.train_path
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_set, data_loader = load_plants_data()


    model = load_model(len(train_set), hyperparameters)
    model.to(device)
    # model.cuda()        # by default will send your model to the "current device"


    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0

    best_model_params = copy.deepcopy(model.state_dict())

    for epoch in range(NUM_EPOCHS):
        print(f'Epoch: {epoch + 1}/{NUM_EPOCHS}')
        print('-' * len(f'Epoch: {epoch + 1}/{NUM_EPOCHS}'))

        training_loss = 0.0
        training_corrects = 0

        for i, (inputs, labels) in enumerate(tqdm(data_loader)):
            # Method 1
            # inputs = Variable(inputs.cuda())
            # labels = Variable(labels.cuda())

            # Method 2
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)         # Resnet18 output
            
            _, preds = torch.max(outputs.data, 1)   
            
            loss = criterion(outputs, labels)       

            optimizer.zero_grad() 
            loss.backward() 
            optimizer.step() 

            training_loss += loss.data * inputs.size(0)
            training_corrects += (preds == labels.data).sum().item() 

        training_loss = training_loss / len(train_set)
        training_acc = training_corrects / len(train_set)

        print(f'Training loss: {training_loss:.4f}\t accuracy: {training_acc:.4f}\n')

        if training_acc > best_acc:
            # Delete previous model with similar accuracy
            if os.path.isfile(f'state_dict-{best_acc:.2f}-best_train_acc.pth'):
                os.remove(f'state_dict-{best_acc:.2f}-best_train_acc.pth')

            best_acc = training_acc
            best_model_params = copy.deepcopy(model.state_dict())

            print(f"Epoch {epoch + 1}/{NUM_EPOCHS}: saving model acc={best_acc:.02f}...")
            torch.save(model.state_dict(), f'state_dict-{best_acc:.2f}-best_train_acc.pth')
        else:
            print(f"Epoch {epoch + 1}/{NUM_EPOCHS}: lower acc, no save acc={training_acc}")

    model.load_state_dict(best_model_params)

    print(f"FINAL saving model acc={best_acc:.02f}...")
    torch.save(model, f'model-{best_acc:.02f}-best_train_acc.pth')

    return model

def train_valid(hyperparameters, train=True, plot=True):
    global NUM_EPOCHS
    global BATCH_SIZE
    global LR
    global NUM_WORKERS
    global PRETRAINED_MODEL

    global TRAIN_PATH

    NUM_EPOCHS = hyperparameters.num_epochs
    BATCH_SIZE = hyperparameters.batch_size
    LR = hyperparameters.lr
    NUM_WORKERS = hyperparameters.num_workers
    PRETRAINED_MODEL = hyperparameters.pretrained_model_path

    TRAIN_PATH = hyperparameters.train_path

    save_every_num_epoch = hyperparameters.save_every_num_epoch

    # create the lists of accuracies/loss
    for file_option in FileOption:
        file_option_list = create_file_option_file([], file_option)
        hyperparameters.set_accuracy_loss_dict(file_option_list, file_option)

    # print out accuracy/loss for training/validation
    hyperparameters.print_accuracy_loss_dict()

    # create model counter
    model_counter = create_model_counter(0)
    hyperparameters.model_counter = model_counter

    # print out model counter (starting epoch)
    hyperparameters.print_model_counter()

    model = None

    if train:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        train_set, valid_set, data_loader, data_loader_valid = load_plant_validation_data()

        model = load_model(len(train_set), hyperparameters)
        model.to(device)
        # model.cuda()        # by default will send your model to the "current device"


        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()

        best_acc = 0.0

        best_model_params = copy.deepcopy(model.state_dict())

        for epoch in range(NUM_EPOCHS):
            print()
            print(f'Epoch: {epoch + 1}/{NUM_EPOCHS}')
            print('-' * len(f'Epoch: {epoch + 1}/{NUM_EPOCHS}'))

            training_loss = 0.0
            training_corrects = 0

            for i, (inputs, labels) in enumerate(tqdm(data_loader)):
                # Method 1
                # inputs = Variable(inputs.cuda())
                # labels = Variable(labels.cuda())

                # Method 2
                inputs, labels = inputs.to(device), labels.to(device)
                #print("inputs: {}".format(inputs))
                #print("labels: {}".format(labels))
                outputs = model(inputs)         # Resnet18 output
                #print("outputs: {}".format(outputs))
                _, preds = torch.max(outputs.data, 1)
                #print("preds: {}".format(preds))
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                training_loss += loss.data * inputs.size(0)
                training_corrects += (preds == labels.data).sum().item()

            training_loss = training_loss / len(train_set)
            training_acc = training_corrects / len(train_set)

            # FIXME: loss is a tensor...
            # save training accuracy/loss
            hyperparameters.add_accuracy_loss_dict_value(FileOption.TRAINING_ACCURACY, training_acc)
            hyperparameters.add_accuracy_loss_dict_value(FileOption.TRAINING_LOSS, training_loss.item())

            print()
            print(f'Training loss: {training_loss:.4f}\t accuracy: {training_acc:.4f}\n')

            # validation
            validate(model, device, valid_set, data_loader_valid, hyperparameters)

            # FIXME: only at the end
            # # increment model counter
            # hyperparameters.increment_model_counter()

            # save every few epochs (speed up training)
            if (epoch + 1) % save_every_num_epoch == 0:
                # if training_acc > best_acc:

                # Delete previous model
                # if os.path.isfile(f'state_dict-{best_acc:.2f}-best_train_acc.pth'):
                #     os.remove(f'state_dict-{best_acc:.2f}-best_train_acc.pth')

                best_acc = training_acc
                best_model_params = copy.deepcopy(model.state_dict())

                print(f"Epoch {epoch + 1}/{NUM_EPOCHS}: saving model acc={best_acc:.02f}...")
                # torch.save(model.state_dict(), f'state_dict-{best_acc:.2f}-best_train_acc_valid.pth')
                # torch.save(model.state_dict(), get_model_dict_filename(best_acc))

                model_counter = hyperparameters.model_counter
                total_epoch = model_counter + (epoch + 1)
                torch.save(model, get_model_filename(best_acc, total_epoch))

        validate_final(model, device, valid_set, data_loader_valid)

        # need to SAVE the accuracy/loss for training/validation
        for file_option in FileOption:
            save_file_option_file(hyperparameters.get_accuracy_loss_dict_list(file_option), file_option)

        # need to SAVE model counter
        model_counter = hyperparameters.model_counter
        total_epoch = model_counter + NUM_EPOCHS
        hyperparameters.model_counter = total_epoch
        save_model_counter(total_epoch)

        # print out accuracy/loss for training/validation
        hyperparameters.print_accuracy_loss_dict()

        # print out model counter
        hyperparameters.print_model_counter()

    if plot:
        # plot the accuracy/loss
        plot_accuracy_loss(hyperparameters)

    # model.load_state_dict(best_model_params)
    #
    # print(f"FINAL saving model acc={best_acc:.02f}...")
    # torch.save(model, f'model-{best_acc:.02f}-best_train_acc_valid.pth')

    return model

def validate_final(model, device, valid_set, data_loader_valid):
    model.eval()

    classes = [_dir.name for _dir in Path(TRAIN_PATH).glob('*')]
    real_class_lst = []
    calc_class_lst = []
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(data_loader_valid)):
            # print(f"Testing...")
            inputs, labels = inputs.to(device), labels.to(device)
            #print("inputs: {}".format(inputs))
            # print("labels of Valid set?: {}".format(labels))
            #real_class.append(labels.flatten().tolist())
            real_class_lst += [j for j in labels.tolist()]
            outputs = model(inputs)         # Resnet18 output
            #print("outputs: {}".format(outputs))
            _, preds = torch.max(outputs.data, 1)
            # print("preds of valid Set?: {}".format(preds))
            #calc_class.append(preds.tolist())
            calc_class_lst += [j for j in preds.tolist()]

           #submission['species'][i] = classes[preds[0]]
    # print(real_class_lst)
    # print(calc_class_lst)
    real_class = np.array(real_class_lst)
    calc_class = np.array(calc_class_lst)
    confusion_matrix(calc_class, real_class)
        #submission.to_csv('submission-fake.csv', index=False)


def plot_accuracy_loss(hyperparameters: HyperParameters):
    plots_dir = "plots"

    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    model_counter = hyperparameters.model_counter

    # plot accuracy
    fig, ax = plt.subplots()

    file_option = FileOption.TRAINING_ACCURACY
    file_option_list = hyperparameters.get_accuracy_loss_dict_list(file_option)
    plt.plot(list(range(len(file_option_list))), file_option_list, '.--', label=file_option.get_file_option_name())

    file_option = FileOption.VALIDATION_ACCURACY
    file_option_list = hyperparameters.get_accuracy_loss_dict_list(file_option)
    plt.plot(list(range(len(file_option_list))), file_option_list, '.--', label=file_option.get_file_option_name())

    plt.xlabel(f"Epoch number")
    plt.ylabel(f"Accuracy")
    plt.title(f"Model Accuracy, batch size={BATCH_SIZE}, lr={LR}, epochs={model_counter}")

    plt.legend()
    plt.grid(True)

    # FIXME: need integers on x-axis...float otherwise
    ax.locator_params(integer=True)

    accuracy_plot_filename = get_model_plot_filename(model_counter, "accuracy")
    plt.savefig(f"{plots_dir}/{accuracy_plot_filename}", bbox_inches="tight")

    # plot loss
    fig, ax = plt.subplots()

    file_option = FileOption.TRAINING_LOSS
    file_option_list = hyperparameters.get_accuracy_loss_dict_list(file_option)
    plt.plot(list(range(len(file_option_list))), file_option_list, '.--', label=file_option.get_file_option_name())

    file_option = FileOption.VALIDATION_LOSS
    file_option_list = hyperparameters.get_accuracy_loss_dict_list(file_option)
    plt.plot(list(range(len(file_option_list))), file_option_list, '.--', label=file_option.get_file_option_name())

    plt.xlabel(f"Epoch number")
    plt.ylabel(f"Loss")
    plt.title(f"Model Loss, batch size={BATCH_SIZE}, lr={LR}, epochs={model_counter}")

    plt.legend()
    plt.grid(True)

    # FIXME: need integers on x-axis...float otherwise
    ax.locator_params(integer=True)

    loss_plot_filename = get_model_plot_filename(model_counter, "loss")
    plt.savefig(f"{plots_dir}/{loss_plot_filename}", bbox_inches="tight")


def validate(model, device, valid_set, data_loader_valid, hyperparameters):
    model.eval()

    classes = [_dir.name for _dir in Path(TRAIN_PATH).glob('*')]
    real_class_lst = []
    calc_class_lst = []

    validation_loss = 0
    validation_corrects = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(data_loader_valid)):
            # print(f"Testing...")
            inputs, labels = inputs.to(device), labels.to(device)
            # print("inputs: {}".format(inputs))
            # print("labels of Valid set?: {}".format(labels))
            # real_class.append(labels.flatten().tolist())
            real_class_lst += [j for j in labels.tolist()]
            outputs = model(inputs)  # Resnet18 output
            # print("outputs: {}".format(outputs))
            _, preds = torch.max(outputs.data, 1)
            # print("preds of valid Set?: {}".format(preds))
            # calc_class.append(preds.tolist())
            calc_class_lst += [j for j in preds.tolist()]

            validation_corrects += (preds == labels.data).sum().item()

            loss = criterion(outputs, labels)
            validation_loss += loss.data * inputs.size(0)

        validation_loss = validation_loss / len(valid_set)
        validation_acc = validation_corrects / len(valid_set)

        # FIXME: loss is a tensor...
        # save training accuracy/loss
        hyperparameters.add_accuracy_loss_dict_value(FileOption.VALIDATION_ACCURACY, validation_acc)
        hyperparameters.add_accuracy_loss_dict_value(FileOption.VALIDATION_LOSS, validation_loss.item())

        print()
        print(f'Validation loss: {validation_loss:.4f}\t accuracy: {validation_acc:.4f}\n')


    # if __name__ == '__main__':
#     train()
