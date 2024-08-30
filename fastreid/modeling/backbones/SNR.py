from torch import nn
from fastreid.layers import SELayer

class SNR(nn.Module):
    def __init__(self, feature):
        super(SNR, self).__init__()
        self.inbn = nn.InstanceNorm2d(feature)
        self.se = SELayer(feature )
    def forward(self,x):
        f = self.inbn( x )
        R = x - f
        R_plus = self.se(R)
        R_minus = R - R_plus

        F_plus = f + R_plus
        F_minus = f + R_minus

        return F_plus, F_minus, f
