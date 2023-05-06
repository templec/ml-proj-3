from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image

class PlantsDataset(Dataset):
    
    def __init__(self,root_dir, transform=None, validation=None,tv_set=None,valid_count=None):
        self.root_dir = Path(root_dir)
        self.img = []
        self.idx = []
        self.transform = transform
        
        if validation:
            if self.root_dir.name == 'train':
                for i, _dir in enumerate(self.root_dir.glob('*')):
                    count = 0
                    for img in _dir.glob('*'):
                        if tv_set == 'v':
                            if count < valid_count:
                                self.img.append(img)
                                self.idx.append(i)
                        if tv_set == 't':
                            if count >= valid_count:
                                self.img.append(img)
                                self.idx.append(i)
                        count += 1
            
            else:
                print('Warning: cannot find folder train')                

        else:
            if self.root_dir.name == 'train':
                for i, _dir in enumerate(self.root_dir.glob('*')):
                    for img in _dir.glob('*'):
                        self.img.append(img)
                        self.idx.append(i)

                print(self.idx)

            else:
                print('Warning: cannot find folder train')

    def __len__(self):
        return len(self.img)
    
    def __getitem__(self, index):
        image = Image.open(self.img[index]).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, self.idx[index]
    
    #def make_validation_set(self)