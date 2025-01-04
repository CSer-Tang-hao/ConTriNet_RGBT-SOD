import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
import cv2
from models.ConTriNet import ConTriNet_R50
from data import test_dataset


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--gpu_id',   type=str, default='6', help='select gpu id')
parser.add_argument('--test_path',type=str, default='',help='test dataset path')
parser.add_argument('--model_path', type=str, default='./Checkpoints/ConTriNet_epoch_best.pth', help='path to the model checkpoint')
opt = parser.parse_args()

dataset_path = opt.test_path

#set device for test
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'USE GPU {opt.gpu_id}' if torch.cuda.is_available() else 'USE CPU')

#load the model
model = ConTriNet_R50(channel=64).to(device)
model.load_state_dict(torch.load(opt.model_path))
model.eval()

#test
test_datasets = ['VT821','VT1000', 'VT5000', 'VT-IMAG']


with torch.no_grad():
    for dataset in test_datasets:
        save_path = f'./Predict_maps/{dataset}/'
        os.makedirs(save_path, exist_ok=True)
            
        image_root = os.path.join(dataset_path, dataset, 'RGB')
        gt_root = os.path.join(dataset_path, dataset, 'GT')
        thermal_root = os.path.join(dataset_path, dataset, 'T')
        test_loader = test_dataset(image_root, gt_root, thermal_root, opt.testsize)

        for i in range(test_loader.size):
            image, gt, thermal, name, image_for_post = test_loader.load_data()
            
            gt      = np.asarray(gt, np.float32)
            gt     /= (gt.max() + 1e-8)
            image   = image.to(device)
            thermal   = thermal.to(device)

            pre_res = model(image,thermal)
            res     = pre_res[0]
            res     = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res     = res.sigmoid().data.cpu().numpy().squeeze()
            res     = (res - res.min()) / (res.max() - res.min() + 1e-8)
            
            save_img_path = os.path.join(save_path, name)
            print(f'save img to: {save_img_path}')
            cv2.imwrite(save_img_path, res * 255)

        print('Test Done!')
