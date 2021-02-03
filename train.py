import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from model import VGG
from config import get_config

cf = get_config()


class LogWriter(SummaryWriter):
    def __init__(self, logdir):
        super(LogWriter, self).__init__(logdir)

    def log_training(self, train_loss, step):
        self.add_scalar('train_loss', train_loss, step)


def to_convert_one_hot(y):
    one_hot = torch.argmax(y.data, dim=1)
    one_hot = one_hot.float()
    one_hot.requires_grad = True
    return one_hot


def train(train_loader, valid_loader, writer):

    model = VGG(batch_size=cf.batch_size)
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=cf.learning_rate)

    for e in range(cf.epoch):
        batch_loss = 0
        for batch_x, batch_y in train_loader:
            y_predict = model(batch_x)
            y_one_hot = to_convert_one_hot(y_predict)

            loss = criterion(y_one_hot, batch_y.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss = loss.item()
            batch_loss += loss
            print(f'batch loss: {loss:.3f}')

        print(f'Epoch #{e}: --- Training loss: {batch_loss/cf.batch_size:.3f}')
        writer.log_training(batch_loss/cf.batch_size, e)

        save_path = './models/chkpt-%d.pt' % (e + 1)
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': e
        }, save_path)
        print("Saved checkpoint to: %s" % save_path)


if __name__ == '__main__':
    import os
    from dataloader import dataloader
    train_loader = dataloader('./dogs-vs-cats/')

    os.makedirs('./logs/', exist_ok=True)
    writer = LogWriter('./logs/')

    train(train_loader, [], writer)
