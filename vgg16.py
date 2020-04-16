import  torch.nn as nn

class VGG16(nn.Module):
    def __init__(self, init_weights=True):
        super(VGG16, self).__init__()
        self.features1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.features2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.features3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))
        self.features4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))
        self.features5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))
        self.features6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))
        self.features7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))
        self.features8 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))
        self.features9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))
        self.features10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))
        self.features11 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))
        self.features12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))
        self.features13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))

        self.avpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        out1 = self.features1(x)
        out2 = self.features2(out1)
        out3 = self.features3(out2)
        out4 = self.features4(out3)
        out5 = self.features5(out4)
        out6 = self.features6(out5)
        out7 = self.features7(out6)
        out8 = self.features8(out7)
        out9 = self.features9(out8)
        out10 = self.features10(out9)
        out11 = self.features11(out10)
        out12 = self.features12(out11)
        out13 = self.features13(out12)
        gap_512_3 = self.avpool(out13)  # GAP
        vgg16_features11 = gap_512_3.view(gap_512_3.size(0), -1)
        out = self.fc(vgg16_features11)
        out = self.sigmoid(out)
        return out

    # initialize every parameters
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def init_vgg16_params(self, vgg16):
        features = list(vgg16.features.children())
        features_block =[self.features1,self.features2,self.features3,self.features4, self.features5,self.features6, self.features7,self.features8,self.features9,self.features10,
                         self.features11, self.features12, self.features13]
        vgg_layers = []
        vgg_bn = []
        for _layer in features:
            if isinstance(_layer, nn.Conv2d):
                vgg_layers.append(_layer)
            elif isinstance(_layer, nn.BatchNorm2d):
                vgg_bn.append(_layer)

        me_layers = []
        me_bn = []
        for block in features_block:
            for _layer in block:
                if isinstance(_layer, nn.Conv2d):
                    me_layers.append(_layer)
                elif isinstance(_layer, nn.BatchNorm2d):
                    me_bn.append(_layer)

        for l1, l2 in zip(vgg_layers, me_layers):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data

        for bn1, bn2 in zip(vgg_bn, me_bn):
            if isinstance(bn1, nn.BatchNorm2d) and isinstance(bn2, nn.BatchNorm2d):
                bn2.weight.data = bn1.weight.data
                bn2.bias.data = bn1.bias.data


