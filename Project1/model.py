import torch.nn as nn

class RobustModel(nn.Module):
    """
    TODO: Implement your model
    """
    def __init__(self):
        super(RobustModel, self).__init__()

        self.in_dim = 28 * 28 * 3
        self.out_dim = 10

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer1.apply(self.init_weights)
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2.apply(self.init_weights)
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3.apply(self.init_weights)
        
        self.layer4 = nn.Sequential(
            nn.Linear(64*3*3, 50),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(50, self.out_dim),
            nn.Dropout(0.5)
            # softmax is included in nn.CrossEntropyLoss
        )
        self.layer4.apply(self.init_weights)
    
    def init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            m.bias.data.fill_(0)
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1) # Flatten
        out = self.layer4(out)
        return out
