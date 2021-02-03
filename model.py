import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsummary import summary as summary_
from config import get_config

config = get_config()


class ConvNet(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ConvNet, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel,
                               out_channels=out_channel,
                               kernel_size=3,
                               stride=1,
                               padding=1)

    def forward(self, x):
        return self.conv(x)


class VGG(nn.Module):
    def __init__(self, batch_size):
        super(VGG, self).__init__()
        self.batch = batch_size
        self.conv1 = ConvNet(3, 64)
        self.conv2 = ConvNet(64, 64)

        self.conv3 = ConvNet(64, 128)
        self.conv4 = ConvNet(128, 128)

        self.conv5 = ConvNet(128, 256)
        self.conv6 = ConvNet(256, 256)
        self.conv7 = ConvNet(256, 256)

        self.conv8 = ConvNet(256, 512)
        self.conv9 = ConvNet(512, 512)
        self.conv10 = ConvNet(512, 512)

        self.conv11 = ConvNet(512, 512)
        self.conv12 = ConvNet(512, 512)
        self.conv13 = ConvNet(512, 512)

        self.maxPooling = nn.MaxPool2d(kernel_size=2, stride=2, dilation=1)

        self.linear1 = nn.Linear(in_features=25088, out_features=4096)
        self.linear2 = nn.Linear(in_features=4096, out_features=4096)
        self.linear3 = nn.Linear(in_features=4096, out_features=2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, input_image):
        # ---------------- block 1 -----------------------
        conv1_output = F.relu(self.conv1(input_image))
        conv2_output = F.relu(self.conv2(conv1_output))
        conv2_maxpool = self.maxPooling(conv2_output)

        # ---------------- block 2 -----------------------
        conv3_output = F.relu(self.conv3(conv2_maxpool))
        conv4_output = F.relu(self.conv4(conv3_output))
        conv4_maxpool = self.maxPooling(conv4_output)

        # ---------------- block 3 -----------------------
        conv5_output = F.relu(self.conv5(conv4_maxpool))
        conv6_output = F.relu(self.conv6(conv5_output))
        conv7_output = F.relu(self.conv7(conv6_output))
        conv7_maxpool = self.maxPooling(conv7_output)

        # ---------------- block 4 -----------------------
        conv8_output = F.relu(self.conv8(conv7_maxpool))
        conv9_output = F.relu(self.conv9(conv8_output))
        conv10_output = F.relu(self.conv10(conv9_output))
        conv10_maxpool = self.maxPooling(conv10_output)

        # ---------------- block 5 -----------------------
        conv11_output = F.relu(self.conv11(conv10_maxpool))
        conv12_output = F.relu(self.conv12(conv11_output))
        conv13_output = F.relu(self.conv13(conv12_output))
        conv13_maxpool = self.maxPooling(conv13_output)

        # ---------------- block 6 -----------------------
        conv_flatten = conv13_maxpool.reshape([self.batch, -1])
        linear1_output = F.relu(self.linear1(conv_flatten))
        dropout_output = self.dropout(linear1_output)
        linear2_output = F.relu(self.linear2(dropout_output))
        dropout_output = self.dropout(linear2_output)
        predict = self.linear3(dropout_output)

        return predict


if __name__ == '__main__':
    vgg_net = VGG(2).to('cpu')

    summary_(vgg_net, (3, 224, 224), batch_size=2)
    random_image = torch.rand([2, 3, 224, 224])
    class_predict = vgg_net(random_image)
    print()
    print(class_predict)
    print(class_predict.shape)

    # batch-normalization 안했을 때, model init을 따로해주면 accuracy가 올라가더라! 이언님
