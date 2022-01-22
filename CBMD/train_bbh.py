import os
import sys
import time
import torch
import numpy
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.utils.data
import torch.nn.parallel
import torch.optim as optim
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.fastest = True
from metrics import get_iou
from torch.utils.data import DataLoader
from data_bbh import get_train, get_test, get_source
from flower import Discriminator, Generator
from vgg19 import Vgg19
import torchvision.models as models

import warnings
warnings.filterwarnings("ignore")  
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--source_path_B',       type=str,   default='PATH'  )
parser.add_argument('--slabel_path_B',       type=str,   default='PATH'  )
parser.add_argument('--source_path_C',       type=str,   default='PATH'  )
parser.add_argument('--slabel_path_C',       type=str,   default='PATH'  )

parser.add_argument('--ts_path_B',           type=str,   default='PATH'  )
parser.add_argument('--tsl_path_B',          type=str,   default='PATH'  )
parser.add_argument('--ts_path_C',           type=str,   default='PATH'  )
parser.add_argument('--tsl_path_C',          type=str,   default='PATH'  )
parser.add_argument('--t_path_B',           type=str,   default='PATH'   )
parser.add_argument('--t_path_C',           type=str,   default='PATH'   )
parser.add_argument('--tt_path_B',           type=str,   default='PATH'   )
parser.add_argument('--tt_path_C',           type=str,   default='PATH'   )

parser.add_argument('--netD',              type=str,   default='' )
parser.add_argument('--netG',              type=str,   default='' )
parser.add_argument('--lambdaSEG',         type=float, default=1.0                 )
parser.add_argument('--lambdaADV',         type=float, default=0.001               )
parser.add_argument('--lrD',               type=float, default=0.0001              )
parser.add_argument('--lrG',               type=float, default=0.0002              )
parser.add_argument('--annealStart',       type=int,   default=0                   )
parser.add_argument('--annealEvery',       type=int,   default=55                  )
parser.add_argument('--epochs',            type=int,   default=100                  )
parser.add_argument('--workers',           type=int,   default=0                   )
parser.add_argument('--BN',                type=int,   default=2                   )
parser.add_argument('--SBN',               type=int,   default=1                   )
parser.add_argument('--TBN',               type=int,   default=6                  )
parser.add_argument('--exp',               type=str,   default='BBH'              ) 
parser.add_argument('--display',           type=int,   default=30                  )
parser.add_argument('--evalIter',          type=int,   default=200                 )
opt = parser.parse_args()

opt.manualSeed = 101
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)
print("Random Seed: ", opt.manualSeed)

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def create_exp_dir(exp):
    try:
        os.makedirs(exp)
        print('Creating exp dir: %s' % exp)
    except OSError:
        pass
    return True
       
def adjust_learning_rate(optimizer, init_lr, epoch):
    lrd = init_lr / epoch 
    old_lr = optimizer.param_groups[0]['lr']
    lr = old_lr - lrd
    if lr < 0: lr = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

'''mse+dice''' 
class NoiseRobustDiceLoss(nn.Module):
    def __init__(self, eps=1e-5):
        super(NoiseRobustDiceLoss, self).__init__()
        self.eps = eps
    def forward(self, pre, gt):
        numerator = torch.abs(pre - gt)
        numerator = torch.pow(numerator, 1.5)
        b, c, h, w = pre.size()
        pre_sum = torch.sum(torch.pow(pre.view(b, c, -1), 2), 2)
        gt_sum = torch.sum(torch.pow(gt.view(b, c, -1), 2), 2)
        numer_sum = torch.sum(numerator.view(b, c, -1), 2)
        denom_sum = pre_sum + gt_sum 
        loss_vector = numer_sum / (denom_sum + self.eps)
        loss = torch.mean(loss_vector)
        return loss

'''edge+dice''' 
class EdgeDiceLoss(nn.Module):
    def __init__(self, a=1, b=3):
        super(EdgeDiceLoss, self).__init__()
        self.a = a
        self.b = b
        self.eps = 2.2204e-16
    def forward(self, pre, gt):
        numerator = torch.pow((pre - gt), 2)
        b, c, h, w = pre.size()
        pre_sum = torch.sum(torch.pow(pre.view(b, c, -1), 2), 2)
        gt_sum = torch.sum(torch.pow(gt.view(b, c, -1), 2), 2)
        numer_sum = torch.sum(numerator.view(b, c, -1), 2)
        denom_sum = pre_sum + gt_sum 
        loss_dice = numer_sum / (denom_sum + self.eps)
        
        pre_e = pre - pre*gt
        gt_e = ((-(gt-1))*pre + gt) - pre*gt
        denominator = torch.abs(pre_e - gt_e)
        loss_mae = denominator / (h*w)

        loss = self.a*torch.mean(loss_dice) + self.b*torch.mean(loss_mae)
        return loss   
# 
vgg = Vgg19()
model_dict = vgg.state_dict()

vgg19 = models.vgg19(pretrained=True)
pretrained_dict = vgg19.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

model_dict.update(pretrained_dict) 
vgg.load_state_dict(model_dict)
vgg.cuda()

for param in vgg.parameters():
    param.requires_grad = False
#      
create_exp_dir(opt.exp)
train_dataset  = get_train(opt.source_path_B, opt.slabel_path_B, opt.source_path_C, opt.slabel_path_C, opt.t_path_B, opt.t_path_C,  256)
source_dataset = get_test(opt.ts_path_B, opt.tsl_path_B, opt.ts_path_C, opt.tsl_path_C)
target_dataset = get_source(opt.tt_path_B, opt.tt_path_C)

train_loader  = DataLoader(dataset=train_dataset,  batch_size=opt.BN,  shuffle=False, num_workers=opt.workers, drop_last=True,  pin_memory=True)
source_loader = DataLoader(dataset=source_dataset, batch_size=opt.SBN, shuffle=False, num_workers=opt.workers, drop_last=False, pin_memory=True)
target_loader = DataLoader(dataset=target_dataset, batch_size=opt.TBN, shuffle=False, num_workers=opt.workers, drop_last=False, pin_memory=True)

# 
trainLogger = open('%s/train.log' % opt.exp, 'w')

netD = Discriminator().cuda()
netG = Generator().cuda()

if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))

netD.train()  
netG.train()
# 
MSE_loss = nn.MSELoss().cuda()
criterionSEG = nn.BCELoss().cuda()
criterionBCE = nn.BCEWithLogitsLoss(reduction='mean').cuda()
criterionNRD = NoiseRobustDiceLoss().cuda()
criterionEDL = EdgeDiceLoss().cuda()
# 
lambdaSEG = opt.lambdaSEG
lambdaADV = opt.lambdaADV
# 
optimizerD = optim.Adam(netD.parameters(), lr = opt.lrD, betas = (0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr = opt.lrG, betas = (0.9, 0.999))

# print('Total-parameter-NetG: %s ' % (get_parameter_number(netG)) )
# print('Total-parameter-NetD: %s ' % (get_parameter_number(netD)) )
best_epoch = {'epoch':0, 'IOU':0}

for epoch in range(opt.epochs):
    
    print()
    Loss_D = 0.0
    Loss_adv = 0.0
    Loss_seg = 0.0
    Loss_mse = 0.0
    Loss_nrd = 0.0
    Loss_edl = 0.0
    real_dis_D = 0.0
    fake_dis_D = 0.0
    start_time = time.time()
    ganIterations = 0

    if epoch+1 > opt.annealStart:  
        adjust_learning_rate(optimizerD, opt.lrD, opt.annealEvery)
        adjust_learning_rate(optimizerG, opt.lrG, opt.annealEvery)
        
    for i, data_train in enumerate(train_loader):
        source_B, slabel_B, source_C, slabel_C, target_B, target_C = data_train 
        source_B, slabel_B, source_C, slabel_C, target_B, target_C = source_B.cuda(), slabel_B.cuda(), source_C.cuda(), slabel_C.cuda(), target_B.cuda(), target_C.cuda()

        for p in netD.parameters():
            p.requires_grad = False      
        netG.zero_grad()
        source_pre, map_B, map_C = netG(source_B, source_C)
        st_label = torch.cat([slabel_B, slabel_C], 1)
        
        # feature_map_B = vgg(map_B[0])
        # feature_map_C  = vgg(map_C[0])  
              
        mse_loss = MSE_loss(map_B, map_C)
        
        seg_loss = criterionSEG(source_pre, st_label)
        
        nrd_loss = criterionNRD(source_pre, st_label)
        
        edl_loss = criterionEDL(source_pre, st_label)

        target_pre,_,_ = netG(target_B, target_C)
        real_G = netD(source_pre)
        fake_G = netD(target_pre)
        real_logit_G = real_G - torch.mean(fake_G)
        fake_logit_G = fake_G - torch.mean(real_G)
        real_loss_G = criterionBCE(real_logit_G, torch.zeros_like(real_logit_G))
        fake_loss_G = criterionBCE(fake_logit_G, torch.ones_like(fake_logit_G))
        adv_loss = lambdaADV * (real_loss_G + fake_loss_G)
       
        total_loss = 0.8*seg_loss + 0.01*mse_loss + 0.5*nrd_loss + 0.0*edl_loss + 0.001*adv_loss
        total_loss.backward(retain_graph=True) 
        
        Loss_mse += mse_loss.item()
        Loss_seg += seg_loss.item()
        Loss_nrd += nrd_loss.item()
        Loss_edl += edl_loss.item()
        Loss_adv += adv_loss.item()
        optimizerG.step()
    
        ganIterations += 1   
        # 
        if ganIterations % opt.display == 0:
            # print('[%d/%d][%d/%d] | total:%f | seg:%f | nrd:%f | edl:%f | mse:%f | adv:%f | real:%f | fake:%f' % (epoch+1,opt.epochs,i+1,len(train_loader),total_loss*opt.BN,Loss_seg*opt.BN,Loss_nrd*opt.BN,Loss_edl*opt.BN,Loss_mse*opt.BN,Loss_adv*opt.BN,real_dis_D,fake_dis_D))
            sys.stdout.flush()
            trainLogger.write('[%d/%d][%d/%d] | total:%f | seg:%f | nrd:%f | edl:%f | mse:%f | adv:%f | real:%f | fake:%f\n' % (epoch+1,opt.epochs,i+1,len(train_loader),total_loss*opt.BN,Loss_seg*opt.BN,Loss_nrd*opt.BN,Loss_edl*opt.BN,Loss_mse*opt.BN,Loss_adv*opt.BN,real_dis_D,fake_dis_D))
            trainLogger.flush()
            Loss_D = 0.0
            Loss_adv = 0.0
            Loss_seg = 0.0
            Loss_mse = 0.0
            Loss_nrd = 0.0
            Loss_edl = 0.0
            real_dis_D = 0.0
            fake_dis_D = 0.0
            
        if ganIterations % opt.evalIter == 0:
            netG.eval()
            with torch.no_grad():
                for k, data_val in enumerate(target_loader):  # target_loader
                    val_target_B, val_target_C = data_val
                    val_target_B, val_target_C = val_target_B.cuda(), val_target_C.cuda()
                    val_target_pre,_,_ = netG(val_target_B, val_target_C)
                    val_target_pre_B = val_target_pre[:,0:1,:,:]
                    val_target_pre_C = val_target_pre[:,1:2,:,:]
                    # vutils.save_image(val_target_B, '%s/s_epoch_%d_iter%06d.png' % (opt.exp, epoch+1, ganIterations), normalize=False, scale_each=False)  # val_target
                    # vutils.save_image(val_target_pre_B, '%s/sl_epoch_%d_iter%06d.png' % (opt.exp, epoch+1, ganIterations), normalize=False, scale_each=False)
                    # vutils.save_image(val_target_C, '%s/t_epoch_%d_iter%06d.png' % (opt.exp, epoch+1, ganIterations), normalize=False, scale_each=False)  # val_target
                    # vutils.save_image(val_target_pre_C, '%s/tl_epoch_%d_iter%06d.png' % (opt.exp, epoch+1, ganIterations), normalize=False, scale_each=False)
            netG.train()   
            
    # print("Epoch: %d Learning rate: D: %f G: %f" % (epoch+1, optimizerD.param_groups[0]['lr'], optimizerG.param_groups[0]['lr']))
    
    # 模型测试
    # print('Model test')
    netG.eval()
    iouss = []
    ioutt = []
    with torch.no_grad():
        for j, data_test in enumerate(source_loader):
            test_source_B, test_label_B, test_target_C, test_label_C  = data_test
            test_source_B, test_label_B, test_target_C, test_label_C = test_source_B.cuda(), test_label_B.cuda(), test_target_C.cuda(), test_label_C.cuda()
            test_pre,_,_ = netG(test_source_B, test_target_C)

            test_source_pre_B = test_pre[:,0:1,:,:]
            test_target_pre_C = test_pre[:,1:2,:,:]
            test_source_pre_B = torch.squeeze(test_source_pre_B).cpu().numpy()
            test_target_pre_C = torch.squeeze(test_target_pre_C).cpu().numpy()
            
            test_label_B = torch.squeeze(test_label_B).cpu().numpy()
            test_label_C = torch.squeeze(test_label_C).cpu().numpy()
            
            iou_B = get_iou(test_label_B, test_source_pre_B)
            iou_C = get_iou(test_label_C, test_target_pre_C) 
            
            iouss.append(iou_B)  
            ioutt.append(iou_C)                                                   
    ious_avg = np.mean(iouss)
    iout_avg = np.mean(ioutt)
    # print('Eval Result: [%d/%d] | IOUS:%f | IOUT:%f' % (epoch+1, opt.epochs, ious_avg, iout_avg))
    sys.stdout.flush()
    trainLogger.write('Eval Result: [%d/%d] | IOUS:%f | IOUT:%f\n' % (epoch+1, opt.epochs, ious_avg, iout_avg) )
    trainLogger.flush()
    netG.train()
    
    # 保存最佳模型
    if ious_avg > best_epoch['IOU']:
        torch.save(netD.state_dict(), '%s/netD_epoch%d.pth' % (opt.exp, epoch+1))
        torch.save(netG.state_dict(), '%s/netG_epoch%d.pth' % (opt.exp, epoch+1))
        best_epoch['IOU'] = ious_avg
        best_epoch['epoch'] = epoch+1 

    total_time = time.time() - start_time
    # print('Total-Time: {:.6f} '.format(total_time))  
    
trainLogger.close()