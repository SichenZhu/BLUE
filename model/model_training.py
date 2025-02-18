import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch.optim as optim
import torch.utils.data

import torch.utils.data as data


class Dataset_Unet(data.Dataset):
    def __init__(self, countData, gtProp, ctGEP, ctDeconv_lst):
        self.countData = countData
        self.gtProp = gtProp
        self.length = self.countData.shape[0]
        self.ctDeconv_lst = ctDeconv_lst
        self.ctGEP = ctGEP

    def __getitem__(self, idx):
        countData = torch.from_numpy(self.countData[idx]).float()
        gtProp = torch.from_numpy(self.gtProp[idx]).float()
        ctGEP = torch.from_numpy(self.ctGEP[idx]).float()
        return countData, gtProp, ctGEP

    def __len__(self):
        return self.length
    
""" Blocks of the U-Net model """

class OneLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.one_linear = nn.Sequential(
            nn.Linear(in_dim, out_dim)
        )

    def forward(self, x):
        return self.one_linear(x)

class LinearRelu(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear_relu = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.linear_relu(x)

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # self.up = nn.Upsample(scale_factor=2, mode='linear')
        # self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        
        self.up = nn.ConvTranspose1d(in_channels, in_channels//2, kernel_size=2, stride=2)  # dim -> 2*dim; channel -> 0.5*channel
        self.conv = DoubleConv(in_channels, out_channels)
        

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Prop(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Prop, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(in_dim, 4096),  # 1024
            nn.SiLU(inplace=True),    # ReLU
            nn.Linear(4096, 1024),    # 1024, 256
            nn.SiLU(inplace=True),    # ReLU
            nn.Linear(1024, 256),     # 256, 64
            nn.SiLU(inplace=True),    # ReLU
            nn.Linear(256, 64),       # 64, out_dim
            nn.SiLU(inplace=True),    # ReLU
            nn.Linear(64, out_dim)
        )

    def forward(self, x):
        return F.softmax(self.fc(x), dim=1)
    


""" Full assembly of the parts to form the complete network """

class UNet(nn.Module):
    def __init__(self, n_in_gene, n_out_gene, n_classes):
        super(UNet, self).__init__()
        self.n_in_gene = n_in_gene
        self.n_out_gene = n_out_gene
        self.n_channels = 1
        self.n_classes = n_classes

        self.feature1 = (OneLinear(n_in_gene, 4096))  # 1024
        self.feature2 = (OneLinear(n_in_gene, 4096))  # 1024

        self.inc = (DoubleConv(1, 16))
        self.down1 = (Down(16, 32))
        self.down2 = (Down(32, 64))
        self.down3 = (Down(64, 128))
        self.down4 = (Down(128, 256))
        self.up1 = (Up(256, 128))
        self.up2 = (Up(128, 64))
        self.up3 = (Up(64, 32))
        self.up4 = (Up(32, n_classes))

        self.outGEP = (LinearRelu(4096, n_out_gene))  # 1024

        self.prop = (Prop(4096, n_classes))  # 1024

    def forward(self, x):
        feat1 = self.feature1(x)                # 1 * n_gene -> 1 * 4096
        x1 = self.inc(feat1.unsqueeze(1))       # -> 16 * 4096
        x2 = self.down1(x1)                     # -> 32 * 2048
        x3 = self.down2(x2)                     # -> 64 * 1024
        x4 = self.down3(x3)                     # -> 128 * 512
        x5 = self.down4(x4)                     # -> 256 * 256

        x_ = self.up1(x5, x4)                   # (256, 256) (128, 512) -> convtrans (128, 512) (128, 512) -> concat(256, 512) -> conv (128, 512)
        x_ = self.up2(x_, x3)                   # (128, 512) (64, 1024) -> convtrans (64, 1024) (64, 1024)
        x_ = self.up3(x_, x2)                   # (64, 1024) (32, 2048)  -> convtrans (32, 2048) (32, 2048)
        x_ = self.up4(x_, x1)                   # (32, 2048) (16, 4096) -> convtrans (16, 4096) (16, 6096) -> concat (32, 4096) -> conv (n_classes, 4096)
        recon = self.outGEP(x_)                 # (n_classes, n_genes)

        feat2 = self.feature2(x)
        prop = self.prop(feat2)
        return prop, recon
    


def evaluate(model, tar_loader, n_celltypes, n_output_genes, device):
    model.eval()
    with torch.no_grad():
        val_loss = 0
        # val_loss1= nn.KLDivLoss(log_target=True, reduction="batchmean")
        val_loss_criterion = nn.MSELoss()
        tar_iter = tqdm(enumerate(tar_loader), total=len(tar_loader))
        for _, batch in tar_iter:
            bulkGEP, gtProp, ctGEP = batch
            bulkGEP = bulkGEP.to(device=device).float()
            gtProp = gtProp.to(device=device).float()
            ctGEP = ctGEP.to(device=device).float()
            prop_pred, ctGEP_pred = model(bulkGEP)

            ctGEP_loss = 0
            for i in range(n_celltypes):
                ctGEP_loss += val_loss_criterion(ctGEP_pred[:, i, :], ctGEP[:, i*n_output_genes:(i+1)*n_output_genes])

            prop_loss = val_loss_criterion(prop_pred, gtProp)
            val_loss += ctGEP_loss + prop_loss
    avg_loss = val_loss.item() / len(tar_loader)
    model.train()
    return prop_loss.item(), ctGEP_loss.item(), avg_loss

def train_model(
        model,
        device,
        train_loader,
        val_loader,
        n_celltypes,
        n_input_genes,
        n_output_genes,
        epochs: int = 5,
        learning_rate: float = 1e-4,
        amp: bool = False,
        weight_decay: float = 0,
        dir_checkpoint: str = None,
        coeff_before_ctGEPLoss = 1,
        coeff_before_propLoss = 100,
        grad_scheduler_stepsize = 100,
        grad_scheduler_gamma = 0.1,

):

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=grad_scheduler_stepsize, gamma=grad_scheduler_gamma)
    # decrease lr in optimizer to 0.1lr every $(step_size) steps
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    # criterion1 = nn.KLDivLoss(log_target=True, reduction="batchmean")
    criterion = nn.MSELoss()

    epoch_loss_lst = []
    val_prop_loss_lst = []
    val_ctGEP_loss_lst = []
    val_avg_loss_lst = []

    for epoch in range(1, epochs + 1):
        # 5. Begin training
        print("Epoch {}/{}".format(epoch, epochs))
        model.train()
        epoch_loss = 0
        src_iter = tqdm(enumerate(train_loader), total=len(train_loader))
        for t, batch in src_iter:
            bulkGEP, gtProp, ctGEP = batch
            # bulkGEP: batch_size by n_gene
            # gtProp: batch_size by n_celltype
            # ctGEP: batch_size by (n_gene*n_celltype)

            bulkGEP = bulkGEP.to(device=device).float()
            gtProp = gtProp.to(device=device).float()
            ctGEP = ctGEP.to(device=device).float()

            with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                prop_pred, ctGEP_pred = model(bulkGEP)

                ctGEP_loss = 0
                for i in range(n_celltypes):
                    ctGEP_loss += criterion(ctGEP_pred[:, i, :], ctGEP[:, i*n_output_genes:(i+1)*n_output_genes])

                prop_loss = criterion(prop_pred, gtProp)
                loss = coeff_before_ctGEPLoss * ctGEP_loss + coeff_before_propLoss * prop_loss

            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            epoch_loss += loss.item()

        epoch_loss_lst.append(epoch_loss / len(train_loader))

        # 6. Evaluate on val_data
        val_prop_loss, val_ctGEP_loss, val_avg_loss = evaluate(model, val_loader, n_celltypes, n_output_genes, device)
        val_prop_loss_lst.append(val_prop_loss)
        val_ctGEP_loss_lst.append(val_ctGEP_loss)
        val_avg_loss_lst.append(val_avg_loss)
        print("Current val loss (total): {:.4f}".format(val_avg_loss))

        scheduler.step()

        # if epoch > 9:
        if val_avg_loss < np.min([np.inf] + val_avg_loss_lst[:-1]):
            torch.save(model.state_dict(), dir_checkpoint + '/model_ep{}.pth'.format(epoch))
            print("Epoch {} saved.".format(epoch))
        elif val_prop_loss < np.min([np.inf] + val_prop_loss_lst[:-1]):
            torch.save(model.state_dict(), dir_checkpoint + '/model_ep{}.pth'.format(epoch))
            print("Epoch {} saved.".format(epoch))
        # elif epoch % 10 == 0:
        #     torch.save(model.state_dict(), dir_checkpoint + '/model_ep{}.pth'.format(epoch))
        #     print("Epoch {} saved.".format(epoch))


    return model, epoch_loss_lst, val_prop_loss_lst, val_ctGEP_loss_lst, val_avg_loss_lst