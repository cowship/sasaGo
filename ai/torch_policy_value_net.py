import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, ch=128):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        h = F.relu(self.bn1(self.c1(x)))
        h = self.bn2(self.c2(h))
        return F.relu(x + h)

class AlphaZeroNet(nn.Module):
    def __init__(self, board_size=9, channels=128, n_blocks=3):
        super().__init__()
        self.s = board_size
        self.stem = nn.Sequential(
            nn.Conv2d(2, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(*[ResBlock(channels) for _ in range(n_blocks)])
        self.p_conv = nn.Conv2d(channels, 2, 1, bias=False)
        self.p_bn = nn.BatchNorm2d(2)
        self.p_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)
        self.v_conv = nn.Conv2d(channels, 1, 1, bias=False)
        self.v_bn = nn.BatchNorm2d(1)
        self.v_fc1 = nn.Linear(board_size * board_size, 128)
        self.v_fc2 = nn.Linear(128, 1)

    def forward(self, x, legal_mask=None):
        h = self.trunk(self.stem(x))
        p = F.relu(self.p_bn(self.p_conv(h)))
        p = p.view(p.size(0), -1)
        logits = self.p_fc(p)
        if legal_mask is not None:
            logits = logits.masked_fill(~legal_mask.bool(), float('-inf'))
        log_probs = F.log_softmax(logits, dim=1)
        v = F.relu(self.v_bn(self.v_conv(h)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.v_fc1(v))
        v = torch.tanh(self.v_fc2(v)).squeeze(1)
        return log_probs, v
