import torch
import torch.nn as nn

class AutoEncoderUnet(nn.Module):
    def __init__(self, chnum_in):
        super(AutoEncoderUnet, self).__init__()
        print('AutoEncoderUnet')

        self.chnum_in = chnum_in
        feature_num = 64
        feature_num_x2 = 128
        feature_num_fc = 9216
        label_num = 10
        num_layers = 4

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels = self.chnum_in, out_channels = feature_num, kernel_size=3),
            nn.BatchNorm2d(feature_num),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels = feature_num, out_channels = feature_num, kernel_size=3),
            nn.BatchNorm2d(feature_num),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5),  
            nn.Flatten(),  
            nn.Linear(feature_num_fc, feature_num_x2),            
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.BatchNorm1d(feature_num_x2), 
            nn.Dropout(0.5),  
            nn.Linear(feature_num_x2, label_num)
        )

    def forward(self, x):
        funcs = []
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        funcs.append(out1)
        funcs.append(out2)
        funcs.append(out3)
        funcs.append(out4)
        
        # print(funcs[1].shape)
        return funcs