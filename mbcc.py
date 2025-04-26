import torch
import torch.nn as nn


class CA(nn.Module):  # one branch
    def __init__(self, input_dim, num_classes, T, lam):
        super(CA, self).__init__()
        self.T = T      # temperature       
        self.lam = lam  # Lambda

        self.head = nn.Conv2d(input_dim, num_classes, 1, bias=False)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        # x (B d H W)
        score = self.head(x) / torch.norm(self.head.weight, dim=1, keepdim=True).transpose(0, 1)  # B, CLASSES, H, W

        score = score.flatten(2)  # score (B C HxW)
        base_logit = torch.mean(score, dim=2)  # B, C

        if self.T == 99:  # max-pooling
            att_logit, _ = torch.max(score, dim=2)
        else:
            score_soft = self.softmax(score * self.T)
            att_logit = torch.sum(score * score_soft, dim=2)

        return base_logit + self.lam * att_logit


class MBCC(nn.Module):

    def __init__(self, lam, input_dim, num_classes, learn=False):
        super(MBCC, self).__init__()
        self.maxhead = CA(input_dim, num_classes, 99, lam)

        self.temp_list = torch.arange(1, num_classes + 1, dtype=torch.float32).cuda()
        if learn:
            self.temp_list = nn.Parameter(self.temp_list)

        self.multi_head = nn.ModuleList([
            CA(input_dim, num_classes, self.temp_list[i], lam)
            for i in range(num_classes - 1)])

    def forward(self, x):
        logit = 0.
        for head in self.multi_head:
            logit += head(x)
        logit += self.maxhead(x)
        return logit

