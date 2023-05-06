import torch
import torch.nn as nn
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

args = parse_args() 

NUM_EPOCHS = args.epochs 
BATCH_SIZE = args.batch_size
LR = args.lr                    
NUM_WORKERS = args.num_workers
TRAIN_PATH = './data/plant-seedlings-classification-cs429529/train'
PRETRAINED_MODEL = args.pretrained_model_path

def load_plants_data():
    
    kwargs = {'num_workers': NUM_WORKERS}

    
    data_transform = transforms.Compose(
        [transforms.RandomResizedCrop(224),
         transforms.ToTensor(),
         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
    )

    train_set = PlantsDataset(root_dir=Path(TRAIN_PATH), transform=data_transform)
    print()
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

    train_set = PlantsDataset(root_dir=Path(TRAIN_PATH), transform=data_transform,validation=True,tv_set='t',valid_count=50)
    valid_set = PlantsDataset(root_dir=Path(TRAIN_PATH), transform=data_transform,validation=True,tv_set='v',valid_count=50)

    data_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE,
                             shuffle=True, **kwargs)
    data_loader_valid = DataLoader(dataset=valid_set, batch_size=BATCH_SIZE,
                             shuffle=True, **kwargs)
    
    return train_set, valid_set, data_loader, data_loader_valid


# Load resnet18 pretrained model
def load_model(num_class):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ResNet
    # model = torch.hub.load('pytorch/vision:v0.5.0',
    #                        'resnet18', pretrained=True)
    # model.fc = nn.Linear(512, num_class)  # by Fan (Not necessary)

    # Another method
    model = resnet18(pretrained=True)

    if args.pretrained_model_path is not None:
        # model.load_state_dict(torch.load(PRETRAINED_MODEL))
        model = torch.load(PRETRAINED_MODEL)
    
    return model

def train():
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_set, data_loader = load_plants_data()


    model = load_model(len(train_set))
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
            #print("inputs: {}".format(inputs))
            print("labels: {}".format(labels))
            outputs = model(inputs)         # Resnet18 output
            #print("outputs: {}".format(outputs))
            _, preds = torch.max(outputs.data, 1)   
            print("preds: {}".format(preds))
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
            # Delete previous modelpa
            if os.path.isfile(f'state_dict-{best_acc:.2f}-best_train_acc.pth'):
                os.remove(f'state_dict-{best_acc:.2f}-best_train_acc.pth')

            best_acc = training_acc
            best_model_params = copy.deepcopy(model.state_dict())

            torch.save(model.state_dict(), f'state_dict-{best_acc:.2f}-best_train_acc.pth')

    model.load_state_dict(best_model_params)

    torch.save(model, f'model-{best_acc:.02f}-best_train_acc.pth')
    return model

def train_valid():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_set, valid_set, data_loader, data_loader_valid = load_plant_validation_data()

    
    model = load_model(len(train_set))
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

        print(f'Training loss: {training_loss:.4f}\t accuracy: {training_acc:.4f}\n')

        if training_acc > best_acc:
            # Delete previous modelpa
            if os.path.isfile(f'state_dict-{best_acc:.2f}-best_train_acc.pth'):
                os.remove(f'state_dict-{best_acc:.2f}-best_train_acc.pth')

            best_acc = training_acc
            best_model_params = copy.deepcopy(model.state_dict())

            torch.save(model.state_dict(), f'state_dict-{best_acc:.2f}-best_train_acc_valid.pth')

    model.load_state_dict(best_model_params)
    validate(model, device, valid_set, data_loader_valid)
    
    torch.save(model, f'model-{best_acc:.02f}-best_train_acc_valid.pth')
    return model

def validate(model, device, valid_set, data_loader_valid):
    model.eval()    
    
    
    classes = [_dir.name for _dir in Path(TRAIN_PATH).glob('*')]
    real_class_lst = []
    calc_class_lst = []
    
    with torch.no_grad():   
        
        for i, (inputs, labels) in enumerate(tqdm(data_loader_valid)):
            inputs, labels = inputs.to(device), labels.to(device)
            #print("inputs: {}".format(inputs))
            print("labels of Valid set?: {}".format(labels))
            #real_class.append(labels.flatten().tolist())
            real_class_lst += [j for j in labels.tolist()]
            outputs = model(inputs)         # Resnet18 output
            #print("outputs: {}".format(outputs))
            _, preds = torch.max(outputs.data, 1)   
            print("preds of valid Set?: {}".format(preds))
            #calc_class.append(preds.tolist())
            calc_class_lst += [j for j in preds.tolist()]

           #submission['species'][i] = classes[preds[0]]
    print(real_class_lst)
    print(calc_class_lst)
    real_class = np.array(real_class_lst)
    calc_class = np.array(calc_class_lst)
    confusion_matrix(calc_class, real_class)
        #submission.to_csv('submission-fake.csv', index=False)

if __name__ == '__main__':
    train()