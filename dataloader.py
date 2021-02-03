import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms

from config import get_config

cf = get_config()


def dataloader(path):
    simple_transform = transforms.Compose([transforms.Resize((224, 224))
                                              , transforms.ToTensor()
                                              , transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    training_transform = transforms.Compose([transforms.Resize((224, 224)),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.RandomRotation(0.2),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.32, 0.406], [0.229, 0.224, 0.225])])

    # 전체 데이터를 batch-size 만큼 가져옴 -> [3d-data, label] list 형태로 저장
    # data: (batch, 3, 224, 224) shape, label: (batch) shape
    train_loader = data.DataLoader(torchvision.datasets.ImageFolder(path+'train/',
                                                                    training_transform),
                                                                    batch_size=cf.batch_size,
                                                                    shuffle=True,
                                                                    num_workers=8)
    '''
    valid_loader = data.DataLoader(torchvision.datasets.ImageFolder(path + 'test1/',
                                                                    simple_transform),
                                                                    batch_size=cf.batch_size,
                                                                    num_workers=8)

    return train_loader, valid_loader
    '''
    return train_loader


if __name__ == '__main__':
    train_loader = dataloader('./dogs-vs-cats/')
    print(len(train_loader))
    for bx, by in train_loader:
        print(bx.shape, by.shape)
