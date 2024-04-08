import math
import torch
from torch import nn
from scipy.special import binom
from utils import accuracy


class LossFunction(nn.Module):

    def __init__(self, nOut, nClasses, margin_v2=2, device="cuda", **kwargs):
        super().__init__()
        self.x_dim = nOut  # number of x feature i.e. output of the last fc layer
        self.nClasses = nClasses  # number of output = class numbers
        self.margin = int(margin_v2)  # m
        self.beta = 100
        self.beta_min = 0
        self.scale = 0.99

        self.device = device  # gpu or cpu
        self.ce = nn.CrossEntropyLoss()

        # Initialize L-Softmax parameters
        # self.weight = nn.Parameter(torch.FloatTensor(nOut, nClasses))
        self.weight = nn.Parameter(torch.randn(nOut, nClasses), requires_grad=True)
        self.divisor = math.pi / self.margin  # pi/m
        self.C_m_2n = torch.Tensor(binom(int(margin_v2), range(0, int(margin_v2) + 1, 2))).to(device)  # C_m{2n}
        self.cos_powers = torch.Tensor(range(self.margin, -1, -2)).to(device)  # m - 2n
        self.sin2_powers = torch.Tensor(range(len(self.cos_powers))).to(device)  # n
        self.signs = torch.ones(int(margin_v2) // 2 + 1).to(device)  # 1, -1, 1, -1, ...
        self.signs[1::2] = -1

    def calculate_cos_m_theta(self, cos_theta):
        sin2_theta = 1 - cos_theta**2
        cos_terms = cos_theta.unsqueeze(1) ** self.cos_powers.unsqueeze(0)  # cos^{m - 2n}
        sin2_terms = (sin2_theta.unsqueeze(1)  # sin2^{n}
                      ** self.sin2_powers.unsqueeze(0))

        cos_m_theta = (self.signs.unsqueeze(0) *  # -1^{n} * C_m{2n} * cos^{m - 2n} * sin2^{n}
                       self.C_m_2n.unsqueeze(0) *
                       cos_terms *
                       sin2_terms).sum(1)  # summation of all terms

        return cos_m_theta

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight.data.t())

    def find_k(self, cos):
        # to account for acos numerical errors
        eps = 1e-7
        cos = torch.clamp(cos, -1 + eps, 1 - eps)
        acos = cos.acos()
        k = (acos / self.divisor).floor().detach()
        return k

    def forward(self, x, label=None):
        # if self.training:
        # assert label is not None

        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.x_dim

        x, w = x, self.weight
        beta = max(self.beta, self.beta_min)
        logit = x.mm(w)
        indexes = range(logit.size(0))
        logit_label = logit[indexes, label]

        # cos(theta) = w * x / ||w||*||x||
        w_label_norm = w[:, label].norm(p=2, dim=0)
        x_norm = x.norm(p=2, dim=1)
        cos_theta_label = logit_label / (w_label_norm * x_norm + 1e-10)

        # equation 7
        cos_m_theta_label = self.calculate_cos_m_theta(cos_theta_label)

        # find k in equation 6
        k = self.find_k(cos_theta_label)

        # f_y_i
        logit_label_updated = (w_label_norm *
                                x_norm *
                                (((-1) ** k * cos_m_theta_label) - 2 * k))
        logit_label_updated_beta = (logit_label_updated + beta * logit[indexes, label]) / (1 + beta)

        logit[indexes, label] = logit_label_updated_beta
        self.beta *= self.scale

        output = self.scale * logit
        
        # print("\nlogit", logit.shape)
        # print("\noutput", output.shape)
        # print("\nlabel", label.shape)

        # calculate loss
        loss    = self.ce(output, label)
        prec1   = accuracy(output.detach(), label.detach(), topk=(1,))[0]
        return loss, prec1

        # loss    = self.ce(logit, label)
        # prec1   = accuracy(logit.detach(), label.detach(), topk=(1,))[0]
        # return loss, prec1

        #     return logit
        # else:
        #     assert label is None
        #     return x.mm(self.weight)

# *** V2
# import math

# import torch
# from torch import nn
# from torch.autograd import Variable
# from scipy.special import binom
# from utils import accuracy
# from scipy.special import binom


# class LossFunction(nn.Module):

#     def __init__(self, nOut, nClasses, margin_2=2, **kwargs):
#         super().__init__()
#         print('Initialised L-Softmax margin %.3f'%(margin_2))
#         self.nOut = nOut
#         self.nClasses = nClasses
#         self.margin = int(margin_2)

#         self.ce = nn.CrossEntropyLoss()

#         self.weight = nn.Parameter(torch.FloatTensor(nOut, nClasses))

#         self.divisor = math.pi / self.margin
#         self.coeffs = binom(margin_2, range(0, margin_2 + 1, 2))
#         self.cos_exps = range(self.margin, -1, -2)
#         self.sin_sq_exps = range(len(self.cos_exps))
#         self.signs = [1]
#         for i in range(1, len(self.sin_sq_exps)):
#             self.signs.append(self.signs[-1] * -1)

#     def reset_parameters(self):
#         nn.init.kaiming_normal(self.weight.data.t())

#     def find_k(self, cos):
#         acos = cos.acos()
#         k = (acos / self.divisor).floor().detach()
#         return k

#     def forward(self, x, label=None):
#         if self.training:
#             assert label is not None
#             logit = x.matmul(self.weight)
#             batch_size = logit.size(0)
#             logit_label = logit[range(batch_size), label]
#             weight_label_norm = self.weight[:, label].norm(p=2, dim=0)
#             x_norm = x.norm(p=2, dim=1)
#             # norm_label_prod: (batch_size,)
#             norm_label_prod = weight_label_norm * x_norm
#             # cos_label: (batch_size,)
#             cos_label = logit_label / (norm_label_prod + 1e-10)
#             sin_sq_label = 1 - cos_label**2

#             num_ns = self.margin//2 + 1
#             # coeffs, cos_powers, sin_sq_powers, signs: (num_ns,)
#             coeffs = Variable(x.data.new(self.coeffs))
#             cos_exps = Variable(x.data.new(self.cos_exps))
#             sin_sq_exps = Variable(x.data.new(self.sin_sq_exps))
#             signs = Variable(x.data.new(self.signs))

#             cos_terms = cos_label.unsqueeze(1) ** cos_exps.unsqueeze(0)
#             sin_sq_terms = (sin_sq_label.unsqueeze(1)
#                             ** sin_sq_exps.unsqueeze(0))

#             cosm_terms = (signs.unsqueeze(0) * coeffs.unsqueeze(0)
#                           * cos_terms * sin_sq_terms)
#             cosm = cosm_terms.sum(1)
#             k = self.find_k(cos_label)

#             ls_label = norm_label_prod * (((-1)**k * cosm) - 2*k)
#             logit[range(batch_size), label] = ls_label

#         output = float(logit.max(1)[1])
#         loss    = self.ce(output, label)
#         prec1   = accuracy(output.detach(), label.detach(), topk=(1,))[0]
#         return loss, prec1 
    
               
#         #     return logit
#         # else:
#         #     assert label is None
#         #     return x.matmul(self.weight)