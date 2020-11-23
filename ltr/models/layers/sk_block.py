# -*- coding: utf-8 -*

import torch
import torch.nn as nn


class SKConv(nn.Module):
    def __init__(self, features, M=2, r=2, G=1, stride=1, L=8):
        super(SKConv, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(features, features, kernel_size=3 + i * 2, stride=stride, padding=1 + i, groups=G),
                nn.BatchNorm2d(features),
                nn.LeakyReLU(0.2)
            ))
        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Linear(d, features)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_U = torch.sum(feas, dim=1)
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v

    def kaiming_normal_initialize(self):
        for conv in self.convs:
            for m in conv.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

        for fc in self.fcs:
            nn.init.kaiming_normal_(fc.weight.data, mode='fan_in')
            if fc.bias is not None:
                fc.bias.data.zero_()


class SKConv3x3(nn.Module):
    def __init__(self, features, M=2, r=2, G=1, stride=1, L=8):
        super(SKConv3x3, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.fc = nn.Linear(features, d)
        self.softmax = nn.Softmax(dim=1)
        self.convs = nn.ModuleList([])
        self.fcs = nn.ModuleList([])
        for i in range(M):
            blocks = nn.ModuleList([])
            for j in range(i + 1):
                blocks.append(nn.Sequential(
                    nn.Conv2d(features, features, kernel_size=3, stride=stride, padding=1, groups=G),
                    nn.BatchNorm2d(features),
                    nn.LeakyReLU(0.2)
                ))
            self.convs.append(blocks)
            self.fcs.append(
                nn.Linear(d, features)
            )

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            convx = x
            for block in conv:
                convx = block(convx)
            fea = convx.unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_U = torch.sum(feas, dim=1)
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v

    def kaiming_normal_initialize(self):
        for conv in self.convs:
            for block in conv:
                for m in block.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                        if m.bias is not None:
                            m.bias.data.zero_()
                    elif isinstance(m, nn.BatchNorm2d):
                        m.weight.data.fill_(1)
                        m.bias.data.zero_()

        for fc in self.fcs:
            nn.init.kaiming_normal_(fc.weight.data, mode='fan_in')
            if fc.bias is not None:
                fc.bias.data.zero_()


class SKConvMeanMax(nn.Module):
    def __init__(self, features, M=2, r=2, G=1, stride=1, L=8):
        super(SKConvMeanMax, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.fc = nn.Linear(features, d)
        self.softmax = nn.Softmax(dim=1)
        self.convs = nn.ModuleList([])
        self.fcs = nn.ModuleList([])
        for i in range(M):
            blocks = nn.ModuleList([])
            for j in range(i + 1):
                blocks.append(nn.Sequential(
                    nn.Conv2d(features, features, kernel_size=3, stride=stride, padding=1, groups=G),
                    nn.BatchNorm2d(features),
                    nn.LeakyReLU(0.2)
                ))
            self.convs.append(blocks)
            self.fcs.append(
                nn.Linear(d, features)
            )

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            convx = x
            for block in conv:
                convx = block(convx)
            fea = convx.unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_U = torch.sum(feas, dim=1)
        fea_s_mean = fea_U.mean(-1).mean(-1)
        fea_s_max = fea_U.max(-1)[0].max(-1)[0]
        # sum mean and max
        fea_s = fea_s_mean + fea_s_max
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v

    def kaiming_normal_initialize(self):
        for conv in self.convs:
            for block in conv:
                for m in block.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                        if m.bias is not None:
                            m.bias.data.zero_()
                    elif isinstance(m, nn.BatchNorm2d):
                        m.weight.data.fill_(1)
                        m.bias.data.zero_()

        for fc in self.fcs:
            nn.init.kaiming_normal_(fc.weight.data, mode='fan_in')
            if fc.bias is not None:
                fc.bias.data.zero_()


class SKConvMeanMaxOnly(nn.Module):
    def __init__(self, features, M=2, r=2, G=1, stride=1, L=8):
        super(SKConvMeanMaxOnly, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.fc = nn.Linear(features, d)
        self.softmax = nn.Softmax(dim=1)
        self.convs = nn.ModuleList([])
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(features, features, kernel_size=3 + i * 2, stride=stride, padding=1 + i, groups=G),
                nn.BatchNorm2d(features),
                nn.LeakyReLU(0.2)
            ))
            self.fcs.append(
                nn.Linear(d, features)
            )

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_U = torch.sum(feas, dim=1)
        fea_s_mean = fea_U.mean(-1).mean(-1)
        fea_s_max = fea_U.max(-1)[0].max(-1)[0]
        # sum mean and max
        fea_s = fea_s_mean + fea_s_max
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v

    def kaiming_normal_initialize(self):
        for conv in self.convs:
            for m in conv.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

        for fc in self.fcs:
            nn.init.kaiming_normal_(fc.weight.data, mode='fan_in')
            if fc.bias is not None:
                fc.bias.data.zero_()


class SKConvMeanMaxResidual(nn.Module):
    def __init__(self, features, M=2, r=2, G=1, stride=1, L=8):
        super(SKConvMeanMaxResidual, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.fc = nn.Linear(features, d)
        self.softmax = nn.Softmax(dim=1)
        self.convs = nn.ModuleList([])
        self.fcs = nn.ModuleList([])
        for i in range(M):
            blocks = nn.ModuleList([])
            for j in range(i + 1):
                blocks.append(nn.Sequential(
                    nn.Conv2d(features, features, kernel_size=3, stride=stride, padding=1, groups=G),
                    nn.BatchNorm2d(features),
                    nn.LeakyReLU(0.2)
                ))
            self.convs.append(blocks)

        for i in range(M + 1):
            self.fcs.append(
                nn.Linear(d, features)
            )

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            convx = x
            for block in conv:
                convx = block(convx)
            fea = convx.unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        identity = x.clone()
        identity = identity.unsqueeze_(dim=1)
        # residual
        feas = torch.cat([feas, identity], dim=1)
        fea_U = torch.sum(feas, dim=1)
        fea_s_mean = fea_U.mean(-1).mean(-1)
        fea_s_max = fea_U.max(-1)[0].max(-1)[0]
        # sum mean and max
        fea_s = fea_s_mean + fea_s_max
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v

    def kaiming_normal_initialize(self):
        for conv in self.convs:
            for block in conv:
                for m in block.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                        if m.bias is not None:
                            m.bias.data.zero_()
                    elif isinstance(m, nn.BatchNorm2d):
                        m.weight.data.fill_(1)
                        m.bias.data.zero_()

        for fc in self.fcs:
            nn.init.kaiming_normal_(fc.weight.data, mode='fan_in')
            if fc.bias is not None:
                fc.bias.data.zero_()


class SKConv3x3Max(nn.Module):
    def __init__(self, features, M=2, r=2, G=1, stride=1, L=8):
        super(SKConv3x3Max, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.fc = nn.Linear(features, d)
        self.softmax = nn.Softmax(dim=1)
        self.convs = nn.ModuleList([])
        self.fcs = nn.ModuleList([])
        for i in range(M):
            blocks = nn.ModuleList([])
            for j in range(i + 1):
                blocks.append(nn.Sequential(
                    nn.Conv2d(features, features, kernel_size=3, stride=stride, padding=1, groups=G),
                    nn.BatchNorm2d(features),
                    nn.LeakyReLU(0.2)
                ))
            self.convs.append(blocks)
            self.fcs.append(
                nn.Linear(d, features)
            )

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            convx = x
            for block in conv:
                convx = block(convx)
            fea = convx.unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_U = torch.sum(feas, dim=1)
        fea_s = fea_U.max(-1)[0].max(-1)[0]
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v

    def kaiming_normal_initialize(self):
        for conv in self.convs:
            for block in conv:
                for m in block.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                        if m.bias is not None:
                            m.bias.data.zero_()
                    elif isinstance(m, nn.BatchNorm2d):
                        m.weight.data.fill_(1)
                        m.bias.data.zero_()

        for fc in self.fcs:
            nn.init.kaiming_normal_(fc.weight.data, mode='fan_in')
            if fc.bias is not None:
                fc.bias.data.zero_()


class SKConvMax(nn.Module):
    def __init__(self, features, M=2, r=2, G=1, stride=1, L=8):
        super(SKConvMax, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.fc = nn.Linear(features, d)
        self.softmax = nn.Softmax(dim=1)
        self.convs = nn.ModuleList([])
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(features, features, kernel_size=3 + i * 2, stride=stride, padding=1 + i, groups=G),
                nn.BatchNorm2d(features),
                nn.LeakyReLU(0.2)
            ))
            self.fcs.append(
                nn.Linear(d, features)
            )

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_U = torch.sum(feas, dim=1)
        fea_s = fea_U.max(-1)[0].max(-1)[0]
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v

    def kaiming_normal_initialize(self):
        for conv in self.convs:
            for m in conv.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

        for fc in self.fcs:
            nn.init.kaiming_normal_(fc.weight.data, mode='fan_in')
            if fc.bias is not None:
                fc.bias.data.zero_()
