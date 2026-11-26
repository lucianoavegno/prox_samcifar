# ResNet-18 para CIFAR: convolucion inicial de 3x3 y sin max-pooling
import torch
import torch.nn as nn
import torch.nn.functional as F
import config as cfg


class Bloque(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.atajo = nn.Sequential()
        if stride != 1 or cin != cout:
            self.atajo = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        out = F.relu(self.b1(self.c1(x)))
        out = self.b2(self.c2(out))
        return F.relu(out + self.atajo(x))


class ResNetCIFAR(nn.Module):
    def __init__(self, bloques=(2, 2, 2, 2), n_clases=10):
        super().__init__()
        self.c1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.b1 = nn.BatchNorm2d(64)
        capas, cin = [], 64
        for i, (n, cout) in enumerate(zip(bloques, (64, 128, 256, 512))):
            for k in range(n):
                capas.append(Bloque(cin, cout, 2 if (k == 0 and i > 0) else 1))
                cin = cout
        self.capas = nn.Sequential(*capas)
        self.fc = nn.Linear(512, n_clases)

    def forward(self, x):
        x = F.relu(self.b1(self.c1(x)))
        x = self.capas(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)


def crear_modelo(nombre, n_clases):
    bl = {"resnet18": (2, 2, 2, 2), "resnet34": (3, 4, 6, 3)}[nombre]
    return ResNetCIFAR(bl, n_clases).to(cfg.DEVICE).to(memory_format=torch.channels_last)
