import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
import wandb


class FineTuner(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, feat_dim: int, weight_vis: float) -> None:
        super().__init__()

        self.feat_dim = feat_dim
        # self.finetune = nn.Sequential(
        #     nn.Linear(input_dim, feat_dim, bias=False),
        #     nn.BatchNorm1d(feat_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(feat_dim, feat_dim, bias=False),
        #     nn.BatchNorm1d(feat_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(feat_dim, output_dim)
        # ) 

        self.finetune = nn.Sequential(
            nn.Conv2d(input_dim, feat_dim, 3, padding=1, bias=True),
            # nn.InstanceNorm2d(feat_dim),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(feat_dim, feat_dim, 5, padding=2, bias=True),
            # nn.InstanceNorm2d(feat_dim),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(feat_dim, feat_dim, 5, padding=2, bias=True),
            # nn.InstanceNorm2d(feat_dim),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(feat_dim, output_dim, 3, padding=1, bias=True),
        )

        self.warp_mse = nn.MSELoss()
        self.feat_mse = nn.MSELoss()
        self.feat_d1_mse = nn.MSELoss()

        self.weight_vis = weight_vis

    def forward(self, x):
        x = self.finetune(x)
        # visibility = x[..., 2:]
        # visibility = torch.sigmoid(visibility)
        # return x[..., :2], visibility

        visibility = x[:, 2:, :, :]
        visibility = torch.sigmoid(visibility)
        return x[:, :2, :, :], visibility
    
    
    def compute_loss(self, xy_pred, xy_warped, visibility, fa_xy, fb_xy_pred):
        
        weight_feat = 5e2

        loss_pose_warp = (visibility * self.warp_mse(xy_pred, xy_warped)).mean()
        loss_feat = weight_feat * (visibility * self.feat_mse(fa_xy, fb_xy_pred)).mean()
        loss_vis = self.weight_vis * (1 - visibility.mean())
        # d1_fa_xy = fa_xy[:, :, 1:] - fa_xy[:, :, :-1]
        # loss_feat_d1 = self.feat_d1_mse(fa_xy[:, :, 1:] - fa_xy[:, :, :-1], fb_xy_pred[:, :, 1:] - fb_xy_pred[:, :, :-1])
        loss_total = loss_pose_warp + loss_feat + loss_vis

        loss = {
            'loss_total': loss_total,
            'loss_pose_warp': loss_pose_warp,
            'loss_feat': loss_feat,
            'loss_vis': loss_vis,
        }

        return loss
    
    def compute_overfitting_loss(self, xy_pred, xy_warped, visibility, fa_xy, fb_xy_pred):
        
        weight_pose_warp = 10
        weight_feat = 1
        self.weight_vis = 0.5

        loss_pose_warp = self.warp_mse(xy_pred, xy_warped)
        loss_feat = (self.feat_mse(fa_xy, fb_xy_pred)).mean()
        loss_vis = 1 - visibility.mean()
        loss_total = weight_pose_warp * loss_pose_warp + loss_feat + self.weight_vis * loss_vis
        # d1_fa_xy = fa_xy[:, :, 1:] - fa_xy[:, :, :-1]
        # loss_feat_d1 = self.feat_d1_mse(fa_xy[:, :, 1:] - fa_xy[:, :, :-1], fb_xy_pred[:, :, 1:] - fb_xy_pred[:, :, :-1])
        

        loss = {
            'loss_total': loss_total,
            'loss_pose_warp': loss_pose_warp,
            'loss_feat': loss_feat,
            'loss_vis': loss_vis,
        }

        return loss
    

class TestOpt():

    def __init__(
            self, 
            matches: torch.Tensor,  
            fa: torch.Tensor, 
            fb: torch.Tensor,
            img_size: tuple,
            finetune_feat_dim: int = 64,
            lr: float = 1e-3,
            total_epochs: int = 1000,
            vis_threshold: float = 0.5,
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ) -> None:
        super().__init__()

        self.matches = matches.cpu().numpy()
        self.fa = fa.unsqueeze(0).to(device)
        self.fb = fb.unsqueeze(0).to(device)
        self.f_dim = fa.shape[1]
        self.img_size = img_size
        
        self.finetuner = FineTuner(
            2, 
            3, 
            finetune_feat_dim,
            weight_vis=10,
        ).to(device)
        self.optimizer = optim.AdamW(self.finetuner.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=total_epochs // 2, gamma=0.5)
        self.total_epochs = total_epochs
        self.device = device
        self.vis_threshold = vis_threshold


    def get_global_matches(self, xy: torch.Tensor):
        
        pos_a = self.matches[:, :2]
        pos_b = self.matches[:, 2:]
        
        H_pred, inliers = cv2.findHomography(
            pos_a,
            pos_b,
            method = cv2.RANSAC,
            confidence = 0.99999,
            ransacReprojThreshold = 3 * 1000 / 480,
        )

        H_pred = torch.from_numpy(np.transpose(H_pred)).float().to(self.device)
        
        bs, h, w, c = xy.shape
        xy = xy.reshape(-1, c)
        xy1 = torch.cat([xy, torch.ones_like(xy[:, :1])], dim=-1)
        xy1_warped = torch.mm(xy1, H_pred)
        xy_warped = xy1_warped[:, :2] / xy1_warped[:, 2:]
        xy_warped = xy_warped.reshape(bs, h, w, c).permute(0, 3, 1, 2)

        return xy_warped
    
    
    def fit(self):
        
        print('Start fitting')
        self.finetuner.train()

        step = 0
        h, w = self.img_size
        gridx, gridy = torch.meshgrid(
            torch.linspace(-1 + 1 / w, 1 - 1 / w, w, device=self.device),
            torch.linspace(-1 + 1 / h, 1 - 1 / h, h, device=self.device),
        )
        xy = torch.stack([gridx, gridy], dim=-1).unsqueeze(0)
        fa_xy = nn.functional.grid_sample(
            self.fa, xy, mode='bilinear', padding_mode='border'
        )
        
        xy_warped = self.get_global_matches(xy)
        xy = xy.permute(0, 3, 1, 2)

        for epoch_id in trange(self.total_epochs):
            
            lr = self.optimizer.param_groups[0]['lr']
            # wandb.log({'lr': lr}, step=step)

            xy_pred, visibility = self.finetuner(xy)
            # xy_pred_grid = xy_pred.reshape(1, w, h, 2)
            fb_xy_pred = nn.functional.grid_sample(
                self.fb, xy_pred.permute(0, 2, 3, 1), mode='bilinear', padding_mode='border'
            )

            loss = self.finetuner.compute_overfitting_loss(xy_pred, xy_warped, visibility, fa_xy, fb_xy_pred)
            # for k, v in loss.items():
                # wandb.log({k: v.item()}, step=step)

            self.optimizer.zero_grad()
            loss['loss_total'].backward()
            self.optimizer.step()
            self.scheduler.step()

            step += 1

        print('final')

    def eval(self, h2: int, w2: int):

        print('Start evaluation')
        self.finetuner.eval()

        with torch.no_grad():
            h, w = self.img_size
            gridx, gridy = torch.meshgrid(
                torch.linspace(-1 + 1 / w, 1 - 1 / w, w, device=self.device),
                torch.linspace(-1 + 1 / h, 1 - 1 / h, h, device=self.device),
            )
            xy = torch.stack([gridx, gridy], dim=-1).unsqueeze(0)
            xy = xy.permute(0, 3, 1, 2)

            xy_pred, visibility = self.finetuner(xy)

            xy = xy.permute(0, 2, 3, 1).reshape(-1, 2)
            xy_pred = xy_pred.permute(0, 2, 3, 1).reshape(-1, 2)
            visibility = visibility.permute(0, 2, 3, 1).reshape(-1, 1)

            valid_mask = visibility > self.vis_threshold
            xy_pred = xy_pred[valid_mask[:, 0], :]
            xy = xy[valid_mask[:, 0], :]

        xy = xy.cpu().numpy()
        xy_pred = xy_pred.cpu().numpy()

        offset = 0.5

        xy_img1 = np.stack([
            w * (xy[:, 0] + 1) / 2, 
            h * (xy[:, 1] + 1) / 2], 
        axis=-1) - offset

        xy_img2 = np.stack([
            w2 * (xy_pred[:, 0] + 1) / 2, 
            h2 * (xy_pred[:, 1] + 1) / 2], 
        axis=-1) - offset

        # H_pred, inliers = cv2.findHomography(
        #     xy_img1,
        #     xy_img2,
        #     method = cv2.RANSAC,
        #     confidence = 0.99999,
        #     ransacReprojThreshold = 3 * min(w2, h2) / 480,
        # )

        return xy_img1, xy_img2
