import os
import logging
from decimal import Decimal
from datetime import datetime
import argparse

import torch
import torch.nn.functional as F
import numpy as np
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn

from models.ConTriNet import ConTriNet_R50
from data import get_loader


# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--epoch',       type=int,   default=100,   help='epoch number')
parser.add_argument('--lr',          type=float, default=5e-5,  help='learning rate')
parser.add_argument('--batchsize',   type=int,   default=16,    help='training batch size')
parser.add_argument('--trainsize',   type=int,   default=352,   help='training dataset size')
parser.add_argument('--clip',        type=float, default=0.5,   help='gradient clipping margin')
parser.add_argument('--lw',          type=float, default=0.001, help='weight')
parser.add_argument('--gpu_id',      type=str,   default='5',   help='train use gpu')

parser.add_argument('--rgb_label_root',      type=str, default='xxxxxx/VT5000/Train/RGB/',       help='the training rgb images root')
parser.add_argument('--thermal_label_root',  type=str, default='xxxxxx/VT5000/Train/T/',         help='the training thermal images root')
parser.add_argument('--gt_label_root',       type=str, default='xxxxxx/RGBT/VT5000/Train/GT/',        help='the training gt images root')

parser.add_argument('--save_path',           type=str, default='./Checkpoints/',    help='the path to save models and logs')


opt = parser.parse_args()

# Set the device for training
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
print(f'USE GPU {opt.gpu_id}')

  
cudnn.benchmark = True

#build the model
model = ConTriNet_R50(channel=64)
    
model.cuda()
params    = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100, eta_min=1e-6)

#set the path
train_image_root = opt.rgb_label_root
train_gt_root    = opt.gt_label_root
train_thermal_root = opt.thermal_label_root

save_path        = opt.save_path


if not os.path.exists(save_path):
    os.makedirs(save_path)

#load data
print('load data...')
train_loader = get_loader(train_image_root, train_gt_root,train_thermal_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
total_step   = len(train_loader)

# Set up logging
logging.basicConfig(filename=save_path+'log.log',format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', level = logging.INFO,filemode='a',datefmt='%Y-%m-%d %I:%M:%S %p')
logging.info("Config")
logging.info('epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};save_path:{}'.format(opt.epoch,opt.lr,opt.batchsize,opt.trainsize,opt.clip,save_path))

#set loss function
CE   = torch.nn.BCEWithLogitsLoss()

step = 0
writer     = SummaryWriter(os.path.join(save_path, 'summary'))
best_mae   = 1
best_epoch = 0


def structure_loss(pred, mask):
    weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))

    pred  = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(2,3))
    union = ((pred+mask)*weit).sum(dim=(2,3))
    wiou  = 1-(inter+1)/(union-inter+1)
    return (wbce+wiou).mean()


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def train(train_loader, model, optimizer, scheduler, epoch, save_path):
    global step
    model.train()
    loss_all=0
    epoch_step=0
    try:
        for i, (images, gts, thermals) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            
            images   = images.cuda() 
            gts      = gts.cuda() 
            thermals   = thermals.cuda() 

            ##
            pre_res  = model(images, thermals)
            
            loss1    = structure_loss(pre_res[0], gts) 
            loss2    = structure_loss(pre_res[1], gts)
            loss3    = structure_loss(pre_res[2], gts) 
            loss4    = structure_loss(pre_res[3], gts)

            loss_seg = loss1 + loss2 + loss3 + loss4

            loss = loss_seg 
            loss.backward()

            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            cur_lr = optimizer.param_groups[0]["lr"]

            step+=1
            epoch_step+=1
            loss_all+=loss.data
            if i % 50 == 0 or i == total_step or i==1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], LR: {:.2e}, Loss1: {:.4f} Loss2: {:0.4f} Loss3: {:0.4f} Loss4: {:0.4f}'.
                    format(datetime.now(), epoch, opt.epoch, i, total_step, Decimal(cur_lr), loss1.data, loss2.data,  loss3.data, loss4.data))
                logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], LR: {:.2e}, Loss1: {:.4f} Loss2: {:0.4f} Loss3: {:0.4f} Loss4: {:0.4f}'.
                    format( epoch, opt.epoch, i, total_step, Decimal(cur_lr), loss1.data, loss2.data, loss3.data, loss4.data))
                
        loss_all/=epoch_step
        logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format( epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        scheduler.step()
        torch.save(model.state_dict(), save_path+'ConTriNet_epoch_{}.pth'.format(epoch))
            
    except KeyboardInterrupt: 
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), os.path.join(save_path, 'ConTriNet_epoch_{}.pth'.format(epoch + 1)))
        print('save checkpoints successfully!')
        raise
    finally:
        # Ensure resources are released
        pass
        
        
 
if __name__ == '__main__':
    print("Start train...")
    
    for epoch in range(1, opt.epoch + 1):

        # train
        train(train_loader, model, optimizer, scheduler, epoch, save_path)
        