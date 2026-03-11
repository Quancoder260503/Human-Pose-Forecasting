import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchvision.models import resnet18, ResNet18_Weights
'''
 a trainer for action classifier, a Resnet + GRU is enough for this task (Conv3D is an idea but the time cannot be varied), 
 furthur more, an idea of a GAN based to detect synthesizes and real pose can be used (only do after all others
 methods are done well as GAN is notoriously difficult to train). 
'''

# Since the input channel size will be quite small, hence apply a custom resnet 
class BasicBlock(nn.Module): 
    def __init__(self, in_channels, out_channels, strides = 1, is_last = False): 
        super(BasicBlock, self).__init__() 
        self.in_channels  = in_channels 
        self.out_channels = out_channels 
        self.is_last      = is_last
        self.strides      = strides
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = strides, padding = 1)
        self.bn1   = nn.BatchNorm2d(out_channels) 
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1, bias = False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential() 

        if strides > 1 or in_channels != out_channels : 
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = strides), 
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = torch.relu(self.bn2(self.conv2(out)))
        out = out + self.downsample(x)
        pre_act = out 
        out = torch.relu(out) 
        if self.is_last : 
           return out, pre_act
        else : 
           return out 
    
class Resnet(nn.Module): 
    def __init__(self, num_blocks, in_channels = 25): 
        super(Resnet, self).__init__()
        self.in_planes = 64 
        self.conv1   = nn.Conv2d(in_channels, self.in_planes, kernel_size = 3, padding = 1, stride = 1)
        self.bn1     = nn.BatchNorm2d(self.in_planes)
        self.layer1  = self.make_layer(64 , num_blocks[0], stride = 1) 
        self.layer2  = self.make_layer(128, num_blocks[1], stride = 2) 
        self.layer3  = self.make_layer(256, num_blocks[2], stride = 2) 
        self.layer4  = self.make_layer(512, num_blocks[3], stride = 2) 
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) 

        for m in self.modules(): 
            if isinstance(m, nn.Conv2d): 
                nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'relu') 
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)) :
                nn.init.constant_(m.weight, 1) 
                nn.init.constant_(m.bias,   0)
                 
    
    def make_layer(self, out_channels, num_blocks, stride): 
        strides = [stride] + [1] * (num_blocks - 1) 
        layers  = []
        for i in range(num_blocks):
            stride = strides[i] 
            layers.append(BasicBlock(self.in_planes, out_channels, stride))
            self.in_planes = out_channels 
        return nn.Sequential(*layers)
    
    def forward(self, img): 
        out = F.relu(self.bn1(self.conv1(img)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out) 
        out = self.layer4(out) 
        out = self.avgpool(out) 
        out = torch.flatten(out, 1) 
        return out 
    
class ResGRU(nn.Module):
    def __init__(self, in_channels = 25, input_dim = 128, hidden_dim = 128, num_classes = 60, num_layers = 1, dropout = 0.1):
        self.in_channels  = 25 
        self.resnet = Resnet(in_channels = in_channels, num_blocks = [2, 2, 2, 2])
        self.encode_fc = nn.Linear(512, input_dim) 
        self.gru     = nn.GRU(
            input_size    = 64,
            hidden_size   = hidden_dim, 
            batch_first   = True,  
            bidirectional = True,  
            dropout       = dropout, 
            num_layers    = num_layers        
        ) 
        self.final_fc     = nn.Linear(128, 64) 

    def __init__(self, confidence_maps): 
        """
        :param confidence maps : tensor of shape (T, 25, 64, 64) 
        :return 
            distribution_vectors : tensor of shape (num_classes)
        """
        

if __name__ == '__main__':
    device = 'cuda' 
    inp = torch.randn(1, 25, 64, 64).to(device) 
    model = Resnet(num_blocks = [2, 2, 2, 2]).to(device) 
    out = model(inp) 
    print(out.shape)

    
