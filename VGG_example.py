from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.datasets import FashionMNIST


class VGG_net(nn.Module):
    def __init__(self, cov_net, num_classes):
        super(VGG_net, self).__init__()

        self.cov_net = cov_net
        self.gap = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(7 * 7 * 512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.cov_net(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def make_layers(cfg, bn=False):
    layers = []
    in_channels = 1
    for m in cfg:
        if m == 'p':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            conv2d = nn.Conv2d(in_channels, out_channels=m, kernel_size=3, stride=1, padding=1)
            if bn is True:  # 如果有bn层，则可以卷积层bias=False
                layers.append(conv2d)
                layers.append(nn.BatchNorm2d(m))
                layers.append(nn.ReLU(True))
            else:
                layers.append(conv2d)
                layers.append(nn.ReLU(True))
            in_channels = m
    return nn.Sequential(*layers)


def init_model(cfg, batch_norm=False):
    model = VGG_net(make_layers(cfg, bn=batch_norm), 10)
    return model


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]
        transforms.Normalize(mean=[.5], std=[.5])  # 标准化至[-1,1]
    ])
    train_data = FashionMNIST(root='fashion-mnist/', train=True,
                              download=True, transform=transform)
    test_data = FashionMNIST(root='fashion-mnist/', train=False,
                             download=True, transform=transform)

    train = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=6)
    test = DataLoader(test_data, batch_size=64, shuffle=True, num_workers=6)

    cfg = [64, 'p', 128, 'p', 256, 256, 'p', 512, 512, 'p', 512, 512, 'p']
    model = init_model(cfg, batch_norm=True)
    if torch.cuda.is_available():
        print('有gpu可用')
        model = model.cuda().half()  # 使用半精度进行训练，有助于节约gpu内存，batch_size可以更大

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, eps=1e-6)
    loss_func = nn.CrossEntropyLoss()

    n_epochs = 20
    i = 0

    for epoch in range(n_epochs):
        for x, y in train:
            if torch.cuda.is_available():
                x = x.cuda().half()
                y = y.cuda().half()

            pred = model(x)
            optimizer.zero_grad()
            loss = loss_func(pred, y.long())
            loss.backward()
            optimizer.step()
            i += 1
            if i % 10 == 0:
                print(epoch, i, loss)

        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test:
                if torch.cuda.is_available():
                    data = data.cuda().half()
                    target = target.cuda().half()
                output = model(data)
                test_loss += F.nll_loss(output, target.long(), size_average=False).item()
                pred = torch.max(output, 1)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test.dataset)
        print('\n Test_set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
              .format(test_loss, correct, len(test.dataset),
                      100. * correct / len(test.dataset)))
