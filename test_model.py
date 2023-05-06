import torch
from utils import parse_args
from torch.autograd import Variable
from torchvision import transforms
from pathlib import Path
import pandas as pd
from PIL import Image
from tqdm import tqdm

args = parse_args()
TEST_PATH = './data/plant-seedlings-classification-cs429529/test'
TRAIN_PATH = './data/plant-seedlings-classification-cs429529/train'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test(model):
    model.eval()    
    test_transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )
    
    classes = [_dir.name for _dir in Path(TRAIN_PATH).glob('*')]

    submission = pd.read_csv('./data/plant-seedlings-classification-cs429529/sample_submission.csv')
    with torch.no_grad():   
        for i, file_name in enumerate(tqdm(submission['file'])):
            img = Image.open(Path(TEST_PATH).joinpath(file_name)).convert(
                'RGB')  
            img = test_transform(img).unsqueeze(0)
            inputs = img.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)   # return the index of the maximum value

            submission['species'][i] = classes[preds[0]]

        submission.to_csv('submission-fake.csv', index=False)

if __name__ == '__main__':
    
    model = torch.load(args.pretrained_model_path)
    test(model)