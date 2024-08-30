import torch
import torch.nn as nn
import torch.nn.functional as F

def learnable_filters(gpu, out_dim, kernel_size):
    group_dim = out_dim // 6
    filter_x_half = torch.zeros(group_dim, 1, kernel_size, kernel_size//2)
    filter_x_half.normal_()
    filter_y_half = torch.zeros(group_dim, 1, kernel_size//2, kernel_size)
    filter_y_half.normal_()
    # nn.init.kaiming_normal_(filter_x_half, mode='fan_out', nonlinearity='relu')
    # nn.init.kaiming_normal_(filter_y_half, mode='fan_out', nonlinearity='relu')
    # filter_x = nn.Conv2d(1, group_dim, kernel_size, padding=kernel_size//2)
    # filter_y = nn.Conv2d(1, group_dim, kernel_size, padding=kernel_size//2)
    if gpu is not None:
        filter_x_half, filter_y_half = filter_x_half.cuda(), filter_y_half.cuda()
    return filter_x_half, filter_y_half

def sobel_filters(gpu):
    filter_x = torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    filter_y = torch.Tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    filter_x = filter_x.unsqueeze(0).unsqueeze(0)
    filter_y = filter_y.unsqueeze(0).unsqueeze(0)
    if gpu is not None:
        filter_x, filter_y = filter_x.cuda(), filter_y.cuda()
    return filter_x, filter_y

def color_ratios(ch, ch_x, ch_y):
    # inputs are tuple of len 3, corresponding to R, G, B
    m_x_list = []
    m_y_list = []
    for i,j in ((0,1), (0,2), (1,2)):
        # m_x_list.append((ch_x[i] * ch[j] - ch_x[j] * ch[i]) / (ch[i] * ch[j] + 1e-1))
        # m_y_list.append((ch_y[i] * ch[j] - ch_y[j] * ch[i]) / (ch[i] * ch[j] + 1e-1))
        diff_x = ch_x[i] * ch[j] - ch_x[j] * ch[i]
        diff_y = ch_y[i] * ch[j] - ch_y[j] * ch[i]
        norm_x = ch[i] * ch[j] + torch.abs(diff_x) + 0.01
        norm_y = ch[i] * ch[j] + torch.abs(diff_y) + 0.01
        m_x_list.append(diff_x / norm_x)
        m_y_list.append(diff_y / norm_y)
    return tuple(m_x_list), tuple(m_y_list)


class CRConv2d(nn.Module):
    # color ratio based convolution
    # reference: Wiley "color in computer vision: fundamentals and applications"
    def __init__(self, learn_filter=False, out_dim=64, kernel_size=7):
        super(CRConv2d, self).__init__()
        self.use_cuda = torch.cuda.is_available()
        self.gpu = torch.cuda.current_device() if self.use_cuda else None
        self.learn_filter = learn_filter
        pixel_mean = [0.485, 0.456, 0.406]
        pixel_std = [0.229, 0.224, 0.225]
        self.pixel_mean = torch.Tensor(pixel_mean).view(1, -1, 1, 1).cuda()
        self.pixel_std = torch.Tensor(pixel_std).view(1, -1, 1, 1).cuda()
        if not learn_filter:
            self.out_dim = 1
            self.filter_x, self.filter_y = sobel_filters(self.gpu)
        else:
            self.out_dim = out_dim
            filter_x_half, filter_y_half = learnable_filters(self.gpu, self.out_dim, kernel_size)
            self.filter_x_half = torch.nn.Parameter(filter_x_half, requires_grad=True)
            self.filter_y_half = torch.nn.Parameter(filter_y_half, requires_grad=True)
        # self.bn = BatchNorm(1)

    def forward(self, batch):
        batch_normalized = (batch - self.pixel_mean) / self.pixel_std
        batch_ch = torch.split(batch_normalized, 1, dim=1)
        batch_ch01 = torch.split(F.avg_pool2d(batch, 2, 2), 1, dim=1)
        batch_ch_x, batch_ch_y = [], []
        if not self.learn_filter:
            filter_x, filter_y = self.filter_x, self.filter_y
        else:
            filter_x_zero = torch.zeros(self.filter_x_half.shape[0], 1, self.filter_x_half.shape[2], 1)
            filter_y_zero = torch.zeros(self.filter_y_half.shape[0], 1, 1, self.filter_y_half.shape[3])
            if self.gpu is not None:
                filter_x_zero, filter_y_zero = filter_x_zero.cuda(), filter_y_zero.cuda()
            filter_x = torch.cat((-torch.abs(self.filter_x_half), filter_x_zero, torch.flip(torch.abs(self.filter_x_half),[3])), dim=3)
            filter_y = torch.cat((-torch.abs(self.filter_y_half), filter_y_zero, torch.flip(torch.abs(self.filter_y_half),[2])), dim=2)
        for c in batch_ch:
            batch_ch_x.append(F.conv2d(input=c, weight=filter_x, stride=2, padding=3))
            batch_ch_y.append(F.conv2d(input=c, weight=filter_y, stride=2, padding=3))
            # batch_ch_x.append(F.conv2d(input=c, weight=filter_x, padding=int(filter_x.shape[2] / 2)))
            # batch_ch_y.append(F.conv2d(input=c, weight=filter_y, padding=int(filter_y.shape[2] / 2)))
            # batch_ch_x.append(self.filter_x(c))
            # batch_ch_y.append(self.filter_y(c))
        m_x_list, m_y_list = color_ratios(batch_ch01, batch_ch_x, batch_ch_y)
        if self.out_dim == 1:
            out = 0
            for m_x, m_y in zip(m_x_list, m_x_list):
                out += torch.pow(m_x, 2) + torch.pow(m_y, 2)
            out = torch.sqrt(out)
            out = torch.clamp(out, max=1.0)
        else:
            cat_list = m_x_list + m_y_list
            out = torch.cat(cat_list, dim=1)
            out = torch.clamp(out, max=2.0, min=-2.0)
            remaining_out_dim = self.out_dim - self.out_dim // 6 * 6
            remaining_channels = torch.zeros(out.size(0), remaining_out_dim, out.size(2), out.size(3)).cuda()
            out = torch.cat((out, remaining_channels), dim=1)
        return out