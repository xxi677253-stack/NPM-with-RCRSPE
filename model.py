"""Exact NPM-Net module definitions used for the released weight state_dict.
The classification and regression heads both include Dropout(0.2).
Dropout contains no state_dict parameters and is therefore defined in code.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleBlock_Lite(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        br = out_ch // 3

        self.b1 = nn.Sequential(
            nn.Conv1d(in_ch, br, 3, padding=1),
            nn.BatchNorm1d(br),
            nn.ReLU()
        )

        self.b2 = nn.Sequential(
            nn.Conv1d(in_ch, br, 7, padding=3),
            nn.BatchNorm1d(br),
            nn.ReLU()
        )

        self.b3 = nn.Sequential(
            nn.Conv1d(in_ch, out_ch - 2 * br, 15, padding=7),
            nn.BatchNorm1d(out_ch - 2 * br),
            nn.ReLU()
        )

    def forward(self, x):
        return torch.cat(
            [self.b1(x), self.b2(x), self.b3(x)],
            dim=1
        )


class DilatedResBlock_Lite(nn.Module):
    def __init__(self, ch):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(ch, ch, 3, padding=2, dilation=2),
            nn.BatchNorm1d(ch),
            nn.ReLU(),

            nn.Conv1d(ch, ch, 3, padding=1),
            nn.BatchNorm1d(ch)
        )

    def forward(self, x):
        return F.relu(x + self.conv(x))


class RSPEModulation(nn.Module):
    def __init__(self, rspe_dim, feat_dim):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(rspe_dim, 32),
            nn.ReLU(),
            nn.Linear(32, feat_dim),
            nn.Sigmoid()
        )

    def forward(self, feat, rspe):
        return feat * self.mlp(rspe)


class FinalModel(nn.Module):
    def __init__(self, num_classes, rspe_dim=20):
        super().__init__()

        self.ms = MultiScaleBlock_Lite(1, 24)
        self.dilated = DilatedResBlock_Lite(24)

        self.down = nn.Sequential(
            nn.Conv1d(24, 48, 3, stride=2, padding=1),
            nn.BatchNorm1d(48),
            nn.ReLU(),

            nn.Conv1d(48, 48, 3, stride=2, padding=1),
            nn.BatchNorm1d(48),
            nn.ReLU()
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.rspe_mod = RSPEModulation(
            rspe_dim,
            48
        )

        self.cls = nn.Sequential(
            nn.Linear(48, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

        self.reg = nn.Sequential(
            nn.Linear(48, 64),
            nn.ReLU(),
            nn.Dropout(0.2),  # Aligned with manuscript Section 3.6
            nn.Linear(64, 1)
        )

    def forward(self, x, rspe):
        x = self.ms(x)
        x = self.dilated(x)
        x = self.down(x)

        feat = self.pool(x).flatten(1)
        feat = self.rspe_mod(feat, rspe)

        return (
            self.cls(feat),
            self.reg(feat).squeeze(-1)
        )


if __name__ == "__main__":
    model = FinalModel(num_classes=3, rspe_dim=20)
    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
