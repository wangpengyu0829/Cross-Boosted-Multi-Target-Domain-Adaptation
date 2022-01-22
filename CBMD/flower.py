import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm  

class Discriminator(nn.Module):
    def __init__(self, ch=32):
        super(Discriminator, self).__init__()

        self.conv1_1 = spectral_norm(nn.Conv2d(2,    ch*1, kernel_size=3, padding=1))
        self.conv1_2 = spectral_norm(nn.Conv2d(ch*1, ch*1, kernel_size=3, stride=2, padding=1))

        self.conv2_1 = spectral_norm(nn.Conv2d(ch*1, ch*2, kernel_size=3, padding=1))
        self.conv2_2 = spectral_norm(nn.Conv2d(ch*2, ch*2, kernel_size=3, stride=2, padding=1))

        self.conv3_1 = spectral_norm(nn.Conv2d(ch*2, ch*4, kernel_size=3, padding=1))
        self.conv3_2 = spectral_norm(nn.Conv2d(ch*4, ch*4, kernel_size=3, stride=2,padding=1))

        self.conv4_1 = spectral_norm(nn.Conv2d(ch*4, ch*8, kernel_size=3, padding=1))
        self.conv4_2 = spectral_norm(nn.Conv2d(ch*8, ch*8, kernel_size=3, stride=2,padding=1))

        self.conv5_1 = spectral_norm(nn.Conv2d(ch*8, ch*16, kernel_size=3, padding=1))
        self.conv5_2 = spectral_norm(nn.Conv2d(ch*16,ch*16, kernel_size=3, stride=2,padding=1))

        self.conv6_1 = spectral_norm(nn.Conv2d(ch*16, 1, kernel_size=3, padding=1))

        self.act = nn.LeakyReLU(negative_slope=0.2)
        self._weight_initialize()

    def _weight_initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.act(x)
        x = self.conv1_2(x)
        x = self.act(x)

        x = self.conv2_1(x)
        x = self.act(x)
        x = self.conv2_2(x)
        x = self.act(x)

        x = self.conv3_1(x)
        x = self.act(x)
        x = self.conv3_2(x)
        x = self.act(x)

        x = self.conv4_1(x)
        x = self.act(x)
        x = self.conv4_2(x)
        x = self.act(x)

        x = self.conv5_1(x)
        x = self.act(x)
        x = self.conv5_2(x)
        x = self.act(x)

        x = self.conv6_1(x)
        return x
    
'''上采样模块'''
class Up_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Up_block, self).__init__()
        
        self.demo = nn.Sequential(
            nn.Conv2d(in_ch, in_ch*4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2), 
            nn.Conv2d(in_ch, out_ch,  kernel_size=3, stride=1, padding=1) ) 
        
    def forward(self, x):
        out = self.demo(x)
        return out

'''上采样模块s'''
class Up_blocks(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Up_blocks, self).__init__()
        
        self.demo = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1)) 
        
    def forward(self, x):
        out = self.demo(x)
        return out 
    
'''下采样模块'''
class Down_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down_block, self).__init__()
        
        self.Rpad = nn.ReflectionPad2d(1)
        self.demo = nn.Sequential(
            nn.Conv2d(in_ch, in_ch,  kernel_size=3, stride=2, padding=0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False) )

    def forward(self, x):
        out = self.demo(self.Rpad(x))
        return out
    
'''通道注意力'''
class CA_block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CA_block, self).__init__() 
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.demo = nn.Sequential(
            nn.Conv2d(channel, channel//reduction, kernel_size=1, stride=1, padding=0),
#            nn.BatchNorm2d(channel//reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel//reduction, channel, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid() )

    def forward(self, x):
        w = self.demo(self.avg_pool(x))
        return x * w

'''残差模块'''
class Res_block(nn.Module):
    def __init__(self, channel):
        super(Res_block, self).__init__() 
        self.demo = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1) )
        self.CA = CA_block(channel)

    def forward(self, x):
        out = x + self.CA(self.demo(x))
        return out


'''残差组'''
class Res_group1(nn.Module):
    def __init__(self, channel):
        super(Res_group1, self).__init__()
        
        self.R1 = Res_block(channel)
        self.R2 = Res_block(channel)
        self.R3 = Res_block(channel)
        self.R4 = Res_block(channel)
        self.R5 = Res_block(channel)
        self.R6 = Res_block(channel)
        self.R7 = Res_block(channel)
        self.R8 = Res_block(channel)
#        self.R9 = Res_block(channel)
#        self.R10 = Res_block(channel)
        self.RR = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        x1  = self.R1(x)
        x2  = self.R2(x1)
        x3  = self.R3(x2)
        x4  = self.R4(x3)
        x5  = self.R5(x4)
        x6  = self.R6(x5)
        x7  = self.R5(x6)
        x8  = self.R6(x7)
#        x9  = self.R5(x8)
#        x10 = self.R6(x9)
        out = self.RR(x8) + x
        return out
    
'''嵌套组'''    
class Res_Model_H(nn.Module):
    def __init__(self, channel):
        super(Res_Model_H, self).__init__()

        self.R1  = Res_group1(channel)
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.R2  = Res_group1(channel)
        self.gamma2 = nn.Parameter(torch.zeros(1))
        self.R3  = Res_group1(channel)
        self.gamma3 = nn.Parameter(torch.zeros(1))
        self.R4  = Res_group1(channel)
        self.gamma4 = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        residual = x
        x1  = self.R1(x)  + self.gamma1*residual
        x2  = self.R2(x1) + self.gamma2*residual
        x3  = self.R3(x2) + self.gamma3*residual
        out = self.R4(x3) + self.gamma4*residual
        return out  


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class Generator(nn.Module):
    def __init__(self, channel = 3):
        super(Generator, self).__init__()  

        self.d0 = Down_block(3, 16)
        self.conv1 = DoubleConv(16, 16) 
        self.d1 = Down_block(16, 32)
        self.conv2 = DoubleConv(32, 32) 
        self.d2 = Down_block(32, 64)
        self.conv3 = DoubleConv(64, 64) 
        self.d3 = Down_block(64, 128)
        
        self.R1 = DoubleConv(16,16)
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.R2 = DoubleConv(32,32)
        self.gamma2 = nn.Parameter(torch.zeros(1))
        self.R3 = DoubleConv(64,64)
        self.gamma3 = nn.Parameter(torch.zeros(1))
        
        self.r1 = DoubleConv(16,16)
        self.batta1 = nn.Parameter(torch.zeros(1))
        self.r2 = DoubleConv(32,32)
        self.batta2 = nn.Parameter(torch.zeros(1))
        self.r3 = DoubleConv(64,64)
        self.batta3 = nn.Parameter(torch.zeros(1))
        
        #-------------------- decoder --------------------
        self.Rr3 = DoubleConv(256,256)
        self.u3 = Up_blocks(256,128)
        
        self.Rr2 = DoubleConv(128,128)
        self.u2 = Up_blocks(128,64)
        
        self.Rr1 = DoubleConv(64,64)
        self.u1 = Up_blocks(64,32)
        
        self.Rr0 = DoubleConv(32,32)
        self.u0 = Up_blocks(32,2)
        self.conv = nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1)
 
    def forward(self, Ax, Bx):  

        xl0_A_d0 = self.d0(Ax)
        xl0_B_d0 = self.d0(Bx)
        xl1_A = self.r1(xl0_A_d0) 
        xl1_B = self.R1(xl0_B_d0) 
        xl1_AA = xl1_A + self.batta1*xl1_B
        xl1_BB = xl1_B + self.gamma1*xl1_A
        
        
        xl2_A_1 = self.conv1(xl1_AA)
        xl2_B_1 = self.conv1(xl1_BB)
        xl2_A_d1 = self.d1(xl2_A_1)
        xl2_B_d1 = self.d1(xl2_B_1)
        xl2_A = self.r2(xl2_A_d1) 
        xl2_B = self.R2(xl2_B_d1) 
        xl2_AA = xl2_A + self.batta2*xl2_B
        xl2_BB = xl2_B + self.gamma2*xl2_A
        
        xl3_A_1 = self.conv2(xl2_AA)
        xl3_B_1 = self.conv2(xl2_BB)
        xl3_A_d2 = self.d2(xl3_A_1)
        xl3_B_d2 = self.d2(xl3_B_1)
        xl4_A = self.r3(xl3_A_d2) 
        xl4_B = self.R3(xl3_B_d2) 
        xl4_AA = xl4_A + self.batta3*xl4_B
        xl4_BB = xl4_B + self.gamma3*xl4_A
        
        xl5_A = self.conv3(xl4_AA)
        xl5_B = self.conv3(xl4_BB)
        xl6_A_d3 = self.d3(xl5_A)  # F(A')特征
        xl6_B_d3 = self.d3(xl5_B)  # F(B')特征
        
        #-------------------- decoder --------------------
        xl7 = torch.cat([xl6_A_d3, xl6_B_d3],1)
        xl8 = self.Rr3(xl7)
        xl8_u3 = self.u3(xl8)
        
        xl9 = self.Rr2(xl8_u3)
        xl9_u2 = self.u2(xl9)
        
        xl10 = self.Rr1(xl9_u2)
        xl10_u1 = self.u1(xl10)
        
        xl11 = self.Rr0(xl10_u1)
        xl12_u0 = self.u0(xl11)
        
        xl12 = self.conv(xl12_u0)
        xl12 = F.sigmoid(xl12)
        # xl12_A = xl12[:, 0:1, :, :]
        # xl12_B = xl12[:, 1:2, :, :]
        return xl12, xl6_A_d3, xl6_B_d3
    


    