import torch
import torch.nn as nn
import numpy as np

class UM_loss(nn.Module):
    def __init__(self, alpha, beta, margin):
        super(UM_loss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.margin = margin
        self.ce_criterion = nn.BCELoss()

    def forward(self, score_act, score_bkg, feat_act, feat_bkg, label):
        loss = {}
        
        label = label / torch.sum(label, dim=1, keepdim=True)

        loss_cls = self.ce_criterion(score_act, label)

        label_bkg = torch.ones_like(label).cuda()
        label_bkg /= torch.sum(label_bkg, dim=1, keepdim=True)
        loss_be = self.ce_criterion(score_bkg, label_bkg)

        loss_act = self.margin - torch.norm(torch.mean(feat_act, dim=1), p=2, dim=1)
        loss_act[loss_act < 0] = 0
        loss_bkg = torch.norm(torch.mean(feat_bkg, dim=1), p=2, dim=1)

        loss_um = torch.mean((loss_act + loss_bkg) ** 2)

        loss_total = loss_cls + self.alpha * loss_um + self.beta * loss_be

        loss["loss_cls"] = loss_cls
        loss["loss_be"] = loss_be
        loss["loss_um"] = loss_um
        loss["loss_total"] = loss_total

        return loss_total, loss

def train(net, loader_iter, optimizer, criterion, logger, step):
    net.train()
    
    _data, _label, _, _, _ = next(loader_iter)

    _data = _data.cuda()
    _label = _label.cuda()

    optimizer.zero_grad()

    score_act, score_bkg, feat_act, feat_bkg, _, _ = net(_data)

    cost, loss = criterion(score_act, score_bkg, feat_act, feat_bkg, _label)

    cost.backward()
    optimizer.step()

    for key in loss.keys():
        logger.log_value(key, loss[key].cpu().item(), step)
