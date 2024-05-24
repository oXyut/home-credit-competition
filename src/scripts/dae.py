import torch
from torch import nn



class DAE(nn.Module):
    def __init__(
            self,
            num_continuous: int,
            num_categorical: int
        ) -> None:
        
        super().__init__()
        self.continuous_emb = nn.ModuleList()
        for _ in range(num_continuous):
            self.continuous_emb.append(
                nn.Sequential(
                    nn.Linear(1, 8),
                    nn.ReLU(),
                )
            )
        
        self.cate_emb = nn.Sequential(
            nn.Embedding(3530, 8),
            nn.ReLU(),
        )
        self.backbone = nn.Sequential(
            nn.Linear(num_continuous*8, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, num_continuous),
        )

        self.mask_regressor = nn.Sequential(
            nn.Linear(num_continuous*8, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, num_continuous),
        )

    def forward(
            self,
            x_cont: torch.Tensor,
            x_cate: torch.Tensor
        ) -> torch.Tensor:
        
        z_cont = torch.cat(
            [self.continuous_emb[i](x_cont[:, [i]].to(torch.float32)) for i in range(x_cont.shape[1])],
            dim=1,
        )
        # print(f'z_cont.shape: {z_cont.shape}')
        # print(z_cont)
        # z_cate = self.cate_emb(x_cate)
        # z_cate = z_cate.flatten(start_dim=1)
        # print(f'z_cate.shape: {z_cate.shape}')
        # print(z_cate)

        # z = torch.cat([z_cont, z_cate], dim=1)
        # print(f'z.shape: {z.shape}')

        logit = self.backbone(z_cont)
        mask = self.mask_regressor(z_cont)
        return logit, mask