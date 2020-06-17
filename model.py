import torch
import torch.nn as nn

class CAS_Module(nn.Module):
    def __init__(self, len_feature, num_classes):
        super(CAS_Module, self).__init__()
        self.len_feature = len_feature
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature, out_channels=2048, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU()
        )

        self.conv_2 = nn.Sequential(
            nn.Conv1d(in_channels=2048, out_channels=num_classes, kernel_size=1,
                      stride=1, padding=0, bias=False)
        )
        self.drop_out = nn.Dropout(p=0.7)

    def forward(self, x):
        # x: (B, T, F)
        out = x.permute(0, 2, 1)
        # out: (B, F, T)
        out = self.conv_1(out)
        features = out.permute(0, 2, 1)
        out = self.drop_out(out)
        out = self.conv_2(out)
        out = out.permute(0, 2, 1)
        # out: (B, T, C + 1)
        return out, features

class BMUE(nn.Module):
    def __init__(self, len_feature, num_classes, num_segments):
        super(BMUE, self).__init__()
        self.len_feature = len_feature
        self.num_classes = num_classes

        self.cas_module = CAS_Module(len_feature, num_classes)

        self.softmax = nn.Softmax(dim=1)

        self.softmax_2 = nn.Softmax(dim=2)

        self.num_segments = num_segments
        self.k_act = num_segments // 8
        self.k_bkg = num_segments // 6

        self.drop_out = nn.Dropout(p=0.7)


    def forward(self, x):
        if self.num_segments != x.shape[1]:
            num_segments = x.shape[1]
            k_act = num_segments // 8
            k_bkg = num_segments // 6
        else:
            k_act = self.k_act
            k_bkg = self.k_bkg

        cas, features = self.cas_module(x)

        feat_magnitudes = torch.norm(features, p=2, dim=2)
        
        select_idx = torch.ones_like(feat_magnitudes).cuda()
        select_idx = self.drop_out(select_idx)

        feat_magnitudes_drop = feat_magnitudes * select_idx

        feat_magnitudes_rev = torch.max(feat_magnitudes, dim=1, keepdim=True)[0] - feat_magnitudes
        feat_magnitudes_rev_drop = feat_magnitudes_rev * select_idx

        idx_act = torch.topk(feat_magnitudes_drop, k_act, dim=1)[1]
        idx_act_feat = idx_act.unsqueeze(2).expand([-1, -1, features.shape[2]])
        
        idx_bkg = torch.topk(feat_magnitudes_rev_drop, k_bkg, dim=1)[1]
        idx_bkg_feat = idx_bkg.unsqueeze(2).expand([-1, -1, features.shape[2]])
        idx_bkg_cas = idx_bkg.unsqueeze(2).expand([-1, -1, cas.shape[2]])
        
        feat_act = torch.gather(features, 1, idx_act_feat)
        feat_bkg = torch.gather(features, 1, idx_bkg_feat)

        score_act = torch.mean(torch.topk(cas, k_act, dim=1)[0], dim=1)
        score_bkg = torch.mean(torch.gather(cas, 1, idx_bkg_cas), dim=1)

        score_act = self.softmax(score_act)
        score_bkg = self.softmax(score_bkg)

        cas_softmax = self.softmax_2(cas)

        
        return score_act, score_bkg, feat_act, feat_bkg, features, cas_softmax

