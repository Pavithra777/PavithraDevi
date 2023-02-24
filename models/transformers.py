import torch.nn as nn
import torch.nn.functional as F

class UltimusBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(UltimusBlock, self).__init__()
		#X*K = 48*48x8 > 8
        self.K = nn.Linear(in_dim, out_dim)
		#X*Q = 48*48x8 > 8
        self.Q = nn.Linear(in_dim, out_dim)
		#X*V = 48*48x8 > 8
        self.V = nn.Linear(in_dim, out_dim)
        self.Out = nn.Linear(out_dim, in_dim)

    def forward(self, x):
        k = self.K(x)
        q = self.Q(x)
        v = self.V(x)

        # AM = SoftMax(QTK)/(8^0.5) = 8*8 = 8
        am = F.softmax(torch.matmul(q, k.t()) / (8 ** 0.5), dim=-1)

        # Z = V*AM = 8*8 > 8
        z = torch.matmul(am, v)

        # Z*Out = 8*8x48 > 48
        return self.Out(z)

class TransformerModel(nn.Module):
    def __init__(self):
        super(CIFAR10Model, self).__init__()
        
        #Add 3 Convolutions to arrive at AxAx48 dimensions (e.g. 32x32x3 | 3x3x3x16 >> 3x3x16x32 >> 3x3x32x48)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 48, 3, padding=1)
        
        # Repeat this Ultimus block 4 times
        self.ultimus1 = UltimusBlock(48, 8)
        self.ultimus2 = UltimusBlock(48, 8)
        self.ultimus3 = UltimusBlock(48, 8)
        self.ultimus4 = UltimusBlock(48, 8)
        
        # final fully connected layer
        self.fc = nn.Linear(48, 10)

    def forward(self, x):
        # pass through convolution layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Apply GAP and get 1x1x48, call this X
        x = F.avg_pool2d(x, kernel_size=x.shape[2])
        x = x.view(x.shape[0], -1)
        
        # pass through ultimus blocks
        x = self.ultimus1(x)
        x = self.ultimus2(x)
        x = self.ultimus3(x)
        x = self.ultimus4(x)
        
        # final FC layer that converts 48 to 10 
        x = self.fc(x)
        return x
