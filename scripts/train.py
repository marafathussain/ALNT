import torch.nn.functional as F
from torch.autograd import Variable
import os
import sys
import tempfile
import shutil
import glob
import logging
import nibabel as nib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import math

import monai
from monai.data import NiftiDataset, create_test_image_3d
from monai.inferers import sliding_window_inference
from monai.metrics import compute_meandice
from monai.visualize.img2tensorboard import plot_2d_or_3d_image
from monai.transforms import \
    Compose, AddChannel, LoadNifti, \
    ScaleIntensity, RandSpatialCrop, \
    ToTensor, CastToType, SpatialPad

def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)

from typing import Optional, Union
import warnings
from monai.networks import one_hot
from monai.utils import MetricReduction


# Defining Dice Loss: Adopted from <https://github.com/MIC-DKFZ/nnUNet/tree/master/nnunet>
class DiceMetric:

    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        mutually_exclusive: bool = False,
        sigmoid: bool = False,
        logit_thresh: float = 0.5,
        reduction: Union[MetricReduction, str] = MetricReduction.MEAN,
    ) -> None:
        super().__init__()
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.mutually_exclusive = mutually_exclusive
        self.sigmoid = sigmoid
        self.logit_thresh = logit_thresh
        self.reduction: MetricReduction = MetricReduction(reduction)

        self.not_nans: Optional[torch.Tensor] = None  # keep track for valid elements in the batch

    def __call__(self, y_pred: torch.Tensor, y: torch.Tensor):

        f = compute_meandice(
            y_pred=y_pred,
            y=y,
            include_background=self.include_background,
            to_onehot_y=self.to_onehot_y,
            mutually_exclusive=self.mutually_exclusive,
            sigmoid=self.sigmoid,
            logit_thresh=self.logit_thresh,
        )

        nans = torch.isnan(f)
        not_nans = (~nans).float()
        f[nans] = 0

        t_zero = torch.zeros(1, device=f.device, dtype=torch.float)

        if self.reduction == MetricReduction.MEAN:
            # 2 steps, first, mean by channel (accounting for nans), then by batch

            not_nans = not_nans.sum(dim=1)
            f = torch.where(not_nans > 0, f.sum(dim=1) / not_nans, t_zero)  # channel average

            not_nans = (not_nans > 0).float().sum()
            f = torch.where(not_nans > 0, f.sum() / not_nans, t_zero)  # batch average

        elif self.reduction == MetricReduction.SUM:
            not_nans = not_nans.sum()
            f = torch.sum(f)  # sum over the batch and channel dims
        elif self.reduction == MetricReduction.MEAN_BATCH:
            not_nans = not_nans.sum(dim=0)
            f = torch.where(not_nans > 0, f.sum(dim=0) / not_nans, t_zero)  # batch average
        elif self.reduction == MetricReduction.SUM_BATCH:
            not_nans = not_nans.sum(dim=0)
            f = f.sum(dim=0)  # the batch sum
        elif self.reduction == MetricReduction.MEAN_CHANNEL:
            not_nans = not_nans.sum(dim=1)
            f = torch.where(not_nans > 0, f.sum(dim=1) / not_nans, t_zero)  # channel average
        elif self.reduction == MetricReduction.SUM_CHANNEL:
            not_nans = not_nans.sum(dim=1)
            f = f.sum(dim=1)  # the channel sum
        elif self.reduction == MetricReduction.NONE:
            pass
        else:
            raise ValueError(f"reduction={self.reduction} is invalid.")

        # save not_nans since we may need it later to know how many elements were valid
        self.not_nans = not_nans

        return f


def compute_meandice(
    y_pred: torch.Tensor,
    y: torch.Tensor,
    include_background: bool = True,
    to_onehot_y: bool = False,
    mutually_exclusive: bool = False,
    sigmoid: bool = False,
    logit_thresh: float = 0.5,
):
    """Computes Dice score metric from full size Tensor and collects average.
    Args:
        y_pred: input data to compute, typical segmentation model output.
            it must be one-hot format and first dim is batch, example shape: [16, 3, 32, 32].
        y: ground truth to compute mean dice metric, the first dim is batch.
            example shape: [16, 1, 32, 32] will be converted into [16, 3, 32, 32].
            alternative shape: [16, 3, 32, 32] and set `to_onehot_y=False` to use 3-class labels directly.
        include_background: whether to skip Dice computation on the first channel of
            the predicted output. Defaults to True.
        to_onehot_y: whether to convert `y` into the one-hot format. Defaults to False.
        mutually_exclusive: if True, `y_pred` will be converted into a binary matrix using
            a combination of argmax and to_onehot.  Defaults to False.
        sigmoid: whether to add sigmoid function to y_pred before computation. Defaults to False.
        logit_thresh: the threshold value used to convert (after sigmoid if `sigmoid=True`)
            `y_pred` into a binary matrix. Defaults to 0.5.
    Returns:
        Dice scores per batch and per class, (shape [batch_size, n_classes]).
    Raises:
        ValueError: sigmoid=True is incompatible with mutually_exclusive=True.
    Note:
        This method provides two options to convert `y_pred` into a binary matrix
            (1) when `mutually_exclusive` is True, it uses a combination of ``argmax`` and ``to_onehot``,
            (2) when `mutually_exclusive` is False, it uses a threshold ``logit_thresh``
                (optionally with a ``sigmoid`` function before thresholding).
    """
    n_classes = y_pred.shape[1]
    n_len = len(y_pred.shape)

    if sigmoid:
        y_pred = y_pred.float().sigmoid()

    if n_classes == 1:
        if mutually_exclusive:
            warnings.warn("y_pred has only one class, mutually_exclusive=True ignored.")
        if to_onehot_y:
            warnings.warn("y_pred has only one channel, to_onehot_y=True ignored.")
        if not include_background:
            warnings.warn("y_pred has only one channel, include_background=False ignored.")
        # make both y and y_pred binary
        y_pred = (y_pred >= logit_thresh).float()
        y = (y > 0).float()
    else:  # multi-channel y_pred
        # make both y and y_pred binary
        if mutually_exclusive:
            if sigmoid:
                raise ValueError("sigmoid=True is incompatible with mutually_exclusive=True.")
            y_pred = torch.argmax(y_pred, dim=1, keepdim=True)
            y_pred = one_hot(y_pred, num_classes=n_classes)
        else:
            y_pred = (y_pred >= logit_thresh).float()
        if to_onehot_y:
            y = one_hot(y, num_classes=n_classes)

    if not include_background:
        y = y[:, 1:] if y.shape[1] > 1 else y
        y_pred = y_pred[:, 1:] if y_pred.shape[1] > 1 else y_pred

    assert y.shape == y_pred.shape, "Ground truth one-hot has differing shape (%r) from source (%r)" % (
        y.shape,
        y_pred.shape,
    )
    y = y.float()
    y_pred = y_pred.float()

    # reducing only spatial dimensions (not batch nor channels)
    reduce_axis = list(range(2, n_len))
    intersection = torch.sum(y * y_pred, dim=reduce_axis) + 1e-10

    y_o = torch.sum(y, reduce_axis)
    y_pred_o = torch.sum(y_pred, dim=reduce_axis)
    denominator = y_o + y_pred_o + 1e-10

    f = torch.where(y_o > 0, (2.0 * intersection) / denominator, torch.tensor(float("nan"), device=y_o.device))
    #f = (2.0 * intersection) / denominator
    return f  # returns array of Dice shape: [batch, n_classes]


# Main function
from monai.losses import DiceLoss
def main():

    # Data with expert annotation
    data_dir = '/home/marafath/scratch/miccai'
    epc = 500
    
    i = 0 #folds 0-4 for 5 folds
    np.random.seed(42)   
    idx_ = np.random.permutation(199)

    clean_images = []
    clean_labels = []
    val_images = []
    val_labels = []
    
    val = idx_[(i*40):(i*40)+40]
    val = val*4
    val_idx = []
    
    for k in range(0,len(val)):
        val_idx.append(val[k])
        val_idx.append(val[k]+1)
        val_idx.append(val[k]+2)
        val_idx.append(val[k]+3)
        
    for case in os.listdir(data_dir): 
        if case[0:5] == 'image':
            tmp = int(case[-10:-7])
            fl = case[-10:-7]
            flag = 0

            for m in range(0, len(val_idx)):
                if tmp == val_idx[m]:
                    val_images.append(os.path.join(data_dir,'image_'+fl+'.nii.gz'))
                    val_labels.append(os.path.join(data_dir,'mask_'+fl+'.nii.gz'))
                    flag = 1
                    break
            if flag == 0 and tmp%4 == 1:
                clean_images.append(os.path.join(data_dir,'image_'+fl+'.nii.gz'))
                clean_labels.append(os.path.join(data_dir,'mask_'+fl+'.nii.gz'))
  
    
    # Data with machine-generated annotation
    data_dir = '/home/marafath/project/marafath/data/iran_miccai'
    images = sorted(glob.glob(os.path.join(data_dir, "*_image_ms.nii.gz")))
    masks = sorted(glob.glob(os.path.join(data_dir, "*_infection.nii.gz")))

    total_ = 1468
    idx = np.random.permutation(total_) 
    best_dice = list()
    noisy_bsize = 4
    alpha = 0.5 # alpha = 1 --> RGS; alpha = 0 --> RGM; alpha = 0.5 --> RGS&M [in paper, this is \lambda]
    
    noisy_images = []
    noisy_labels = []
    for pat in range(0,total_):
        noisy_images.append(images[idx[pat]])
        noisy_labels.append(masks[idx[pat]])

    # Defining Transform for noisy and clean data
    train_imtrans = Compose([
        ScaleIntensity(),
        AddChannel(),
        CastToType(), 
        RandSpatialCrop((96, 96, 96), random_size=False),
        SpatialPad((96, 96, 96), mode='constant'),
        ToTensor()
    ])
    train_segtrans = Compose([
        AddChannel(),
        CastToType(), 
        RandSpatialCrop((96, 96, 96), random_size=False),
        SpatialPad((96, 96, 96), mode='constant'),
        ToTensor()
    ])
    
    # Defining Transform for validation data
    val_imtrans = Compose([
        ScaleIntensity(),
        AddChannel(),
        CastToType(),
        SpatialPad((96, 96, 96), mode='constant'),
        ToTensor()
    ])
    val_segtrans = Compose([
        AddChannel(),
        CastToType(),
        SpatialPad((96, 96, 96), mode='constant'),
        ToTensor()
    ])
        
    # Creating a clean training data loader
    clean_ds = NiftiDataset(clean_images, clean_labels, transform=train_imtrans, seg_transform=train_segtrans)
    clean_loader = DataLoader(clean_ds, batch_size=4, shuffle=True, num_workers=8, pin_memory=torch.cuda.is_available())

    # Create a noisy data loader
    noisy_ds = NiftiDataset(noisy_images, noisy_labels, transform=train_imtrans, seg_transform=train_segtrans)
    noisy_loader = DataLoader(noisy_ds, batch_size=noisy_bsize, shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())

    # Create a validation data loader
    val_ds = NiftiDataset(val_images, val_labels, transform=val_imtrans, seg_transform=val_segtrans)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())

        
    # Defining model and hyperparameters
    device = torch.device('cuda:0')
    model = monai.networks.nets.UNet(
        dimensions=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)

    # model.load_state_dict(torch.load('/home/marafath/scratch/saved_model_iA/mic_f_{}.pth'.format(i))) # fine tuning only

    # Starting a typical PyTorch training
    val_interval = 1
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    writer = SummaryWriter()
    for epoch in range(epc):
        print('-' * 10)
        print('epoch {}/{}'.format(epoch + 1, epc))
        model.train()
        epoch_loss = 0
        step = 0

        for noisy_batch_data in noisy_loader:
            step_n = 0
            clean_cycle_loss = 0
            flag = 0
            for clean_batch_data in clean_loader:
                step += 1
                step_n += 1
                meta_net = monai.networks.nets.UNet(
                    dimensions=3,
                    in_channels=1,
                    out_channels=2,
                    channels=(16, 32, 64, 128, 256),
                    strides=(2, 2, 2, 2),
                    num_res_units=2,
                )
                meta_net.load_state_dict(model.state_dict())
                meta_net.to(device)
                meta_optimizer = torch.optim.Adam(meta_net.parameters(), 1e-4)

                inputs_n, labels_n = to_var(noisy_batch_data[0],requires_grad=False), to_var(noisy_batch_data[1], requires_grad=False)
                y_n_hat  = meta_net(inputs_n)
                labels_n = labels_n.type(torch.float)

                ## Assigning one (Dice + CE) loss per batch item, i.e. image
                labels_n_ce = labels_n.long()
                loss_ce = F.cross_entropy(y_n_hat.squeeze(),labels_n_ce.squeeze(), reduce=False)
                tmp = torch.mean(loss_ce,1)
                tmp = torch.mean(tmp,1)
                loss_ce = torch.mean(tmp,1)

                dice = DiceLoss(to_onehot_y=True, softmax=True, reduction='none')
                loss_dc = dice(y_n_hat,labels_n)
                loss_dc = torch.mean(loss_dc,1)
                
                cost = loss_ce + loss_dc
                g_x = to_var(torch.zeros(cost.size()))
                for b_ in range(noisy_bsize):
                    tmp_cost = cost[b_]
                    tmp_cost.backward(retain_graph=True)
                    tmp_grad = []
            
                    for name, param in meta_net.named_parameters():
                        if name == 'model.2.1.conv.unit0.conv.weight': # last layer
                            tmp_grad.append(param.grad.view(-1))
                    tmp_grad = torch.cat(tmp_grad)
                    s_ = Variable(torch.sum(torch.abs(tmp_grad)))
                    n_ = tmp_grad.shape[0]
                    with torch.no_grad():
                        g_x[b_] = s_/n_     
                        
                eps = to_var(torch.zeros(cost.size()))
                l_n_meta = torch.sum(cost * eps)
                
                grads_n = torch.autograd.grad(l_n_meta, (meta_net.parameters()), create_graph=True) # for later use

                meta_net.zero_grad()
                l_n_meta.backward() # gradient --> Grad
                meta_optimizer.step() # parameter update: theta_i+1 = theta_i - \alpha*Grad

                # 2nd forward pass and getting the gradients with respect to epsilon
                inputs_c, labels_c = to_var(clean_batch_data[0],requires_grad=False), to_var(clean_batch_data[1], requires_grad=False)

                y_c_hat = meta_net(inputs_c)
                labels_c = labels_c.type(torch.float)
                
                ## Assigning one (Dice + CE) loss per batch item, i.e. image
                labels_c_ce = labels_c.long()
                loss_ce_c = F.cross_entropy(y_c_hat.squeeze(), labels_c_ce.squeeze())

                dice = DiceLoss(to_onehot_y=True, softmax=True)
                loss_dc_c = dice(y_c_hat, labels_c)
                
                l_c_meta = loss_ce_c + loss_dc_c
                
                grads_c = torch.autograd.grad(l_c_meta, (meta_net.parameters()), only_inputs=True)
                grad_eps = torch.autograd.grad(grads_n, eps, grads_c, only_inputs=True)[0]

                if flag == 0:
                    temp_grad = grad_eps
                    flag = 1
                    break
                else:
                    temp_grad = temp_grad + grad_eps
                    
            temp_grad_w = temp_grad/step_n       
            
            # Computing and normalizing the weights
            w_tilde = torch.clamp(temp_grad_w, min=0)
            norm_c = torch.sum(w_tilde)
            if norm_c.item() == 0:
                w_e = w_tilde  
            else:
                w_e = w_tilde/norm_c
                   
            norm_c_ = torch.sum(g_x)
            if norm_c_.item() == 0:
                w_x = g_x  
            else:
                w_x = g_x/norm_c_
            
            w = torch.empty(noisy_bsize, requires_grad=False)
            w = alpha*w_e + (1-alpha)*w_x   
            w[w < (1/noisy_bsize)] = 0
            
            
            # Computing the loss with the computed weights and then perform a parameter update
            y_n_hat = model(inputs_n)

            ## Assigning one (Dice + CE) loss per batch item, i.e. image
            labels_n_ce = labels_n.long()
            loss_ce = F.cross_entropy(y_n_hat.squeeze(),labels_n_ce.squeeze(), reduce=False)
            tmp = torch.mean(loss_ce,1)
            tmp = torch.mean(tmp,1)
            loss_ce = torch.mean(tmp,1)

            dice = DiceLoss(to_onehot_y=True, softmax=True, reduction='none')
            loss_dc = dice(y_n_hat,labels_n)
            loss_dc = torch.mean(loss_dc,1)

            cost = loss_ce + loss_dc
            l_n = torch.sum(cost * w)

            optimizer.zero_grad()
            l_n.backward()
            optimizer.step()

            clean_cycle_loss += l_n.item()
            epoch_loss += l_n.item()
            
            epoch_len = len(noisy_ds) // noisy_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {clean_cycle_loss:.4f}")
            
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print('epoch {} average loss: {:.4f}'.format(epoch + 1, epoch_loss))

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                metric_sum = 0.
                metric_count = 0
                val_images_ = None
                val_labels_ = None
                val_outputs = None
                cd_sum = []
                for val_data in val_loader:
                    val_images_, val_labels_ = val_data[0].to(device), val_data[1].to(device)
                    roi_size = (160, 160, 96)
                    sw_batch_size = 4
                    val_outputs = sliding_window_inference(val_images_, roi_size, sw_batch_size, model)
                    value = compute_meandice(y_pred=val_outputs, y=val_labels_, include_background=False,
                                     to_onehot_y=True, sigmoid=False, mutually_exclusive=True)
                    v_ = value.cpu()
                    v_ = v_.numpy()
                    cd_sum.append(v_[0][0])
                metric = np.mean(cd_sum)
                if metric > best_metric:
                    best_metric = metric
                    best_dice.append(best_metric)
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), '/home/marafath/scratch/saved_model_iA/mic_ssal_gx_ge_f{}.pth'.format(i))
                    print('saved new best metric model')
                print('current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}'.format(
                    epoch + 1, metric, best_metric, best_metric_epoch))
                writer.add_scalar('val_mean_dice', metric, epoch + 1)
                # plot the last model output as GIF image in TensorBoard with the corresponding image and label
                plot_2d_or_3d_image(val_images_, epoch + 1, writer, index=0, tag='image')
                plot_2d_or_3d_image(val_labels_, epoch + 1, writer, index=0, tag='label')
                plot_2d_or_3d_image(val_outputs, epoch + 1, writer, index=0, tag='output')

    print('train completed, best_metric: {:.4f} at epoch: {}'.format(best_metric, best_metric_epoch))
    writer.close()

if __name__ == '__main__':
    main()