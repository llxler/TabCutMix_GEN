"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import torch
import numpy as np
from scipy.stats import betaprime
import torch.nn as nn
import pandas as pd
#----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

randn_like=torch.randn_like

#SIGMA_MIN=0.002
#SIGMA_MAX=80
#rho=7
#S_churn= 1
#S_min=0
#S_max=float('inf')
#S_noise=1
SIGMA_MIN = torch.tensor(0.002, requires_grad=True)
SIGMA_MAX = torch.tensor(80.0, requires_grad=True)
rho = torch.tensor(7.0, requires_grad=True)
S_churn = torch.tensor(1.0, requires_grad=True)
S_min = torch.tensor(0.0, requires_grad=True)
S_max = torch.tensor(float('inf'), requires_grad=True)  # 注意，这里使用了无穷大表示
S_noise = torch.tensor(1.0, requires_grad=True)




# 定义一个简单的神经网络来估计 log p(y | x_t)
class ConditionNet(nn.Module):
    def __init__(self, input_dim, condition_dim):
        super(ConditionNet, self).__init__()
        self.fc1 = nn.Linear(input_dim + condition_dim, 128)
        self.fc2 = nn.Linear(128, 1)  # 输出 log p(y | x_t)

    def forward(self, x_t, y_condition):
        x_condition = torch.cat([x_t, y_condition], dim=-1)
        x = torch.relu(self.fc1(x_condition))  # ReLU 激活
        log_p = self.fc2(x)
        return log_p

def compute_condition_gradient(x_hat, y_condition, condition_net):
    """
    计算条件梯度

    Args:
    - x_hat (Tensor): 当前带噪声的样本
    - y_condition (Tensor): 条件信息，可能是一个标签或者其他信息
    - condition_net (nn.Module): 条件神经网络，用来计算条件梯度

    Returns:
    - condition_gradient (Tensor): 计算得到的条件梯度
    """
    # 计算条件梯度（根据条件神经网络）
    condition_gradient = condition_net(x_hat, y_condition)
    return condition_gradient

def sample(net, num_samples, dim, num_steps = 50, device = 'cuda:0'):
    #num_steps = torch.tensor(50, dtype=torch.float32, requires_grad=True)

    #latents = torch.randn([num_samples, dim], device=device)
    latents = torch.randn([num_samples, dim], device = 'cpu', requires_grad=True)

    #step_indices = torch.arange(num_steps, dtype=torch.float32, device=latents.device)
    step_indices = torch.arange(num_steps, dtype=torch.float32, device=latents.device, requires_grad=True)

    sigma_min = max(SIGMA_MIN, net.sigma_min)
    sigma_max = min(SIGMA_MAX, net.sigma_max)

    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
                sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho

    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    x_next = latents.to(torch.float32) * t_steps[0]

    num_steps = torch.tensor(50, dtype=torch.float32, requires_grad=True)
    #with torch.no_grad():
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):

        #t_next = t_next.clone().detach().requires_grad_(True)
        #print(num_steps.requires_grad)
        #exit(1)

        x_next = sample_step(net, num_steps, i, t_cur, t_next, x_next)

    return x_next

def sample_step(net, num_steps, i, t_cur, t_next, x_next):

    x_cur = x_next
    # Increase noise temporarily.
    #print(S_churn.requires_grad)
    #print(num_steps.requires_grad)

    #exit(1)
    gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
    #print(t_cur.requires_grad)
    #print(num_steps.requires_grad)
    #exit(1)
    t_hat = net.round_sigma(t_cur + gamma * t_cur)

    #print(t_hat)
    #print(t_cur)
    #exit(1)
    x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
    x_hat = torch.tensor(x_hat.detach().cpu().numpy(), requires_grad=True)


    #x_hat = torch.randn(11097, 72, requires_grad=True)
    #print(x_hat.device)
    #exit(1)

    #exit(1)
    # Euler step.

    #print(x_hat.shape)
    #exit(1)

    # 读取 X_cat_train.npy 文件
    #x_hat = x_hat.detach().requires_grad_(True)
    df = pd.read_csv('E:/TabCutMix/data/shoppers/train.csv')

    # 获取最后一列
    last_column = df.iloc[:, -1]

    # 将True替换为1，False替换为0
    last_column = last_column.replace({True: 1, False: 0})
    # 将列转换为 NumPy 数组
    last_column = last_column.values

    # 将一维数组转换为形状为 [11097, 1] 的张量
    last_column_tensor = torch.tensor(last_column).reshape(11097, 1)

    # 将 y_condition 转换为浮点型并设置 requires_grad
    y_condition = last_column_tensor.float().requires_grad_(True)
    #y_condition = y_condition.to('cuda')
    #torch.set_grad_enabled(True)
    #last_two_columns = X_cat_train[:, -2:]
    #print(last_two_columns)
    #exit(1)

    denoised = net(x_hat, t_hat).to(torch.float32)

    d_cur = (x_hat - denoised) / t_hat

    # 初始化模型
    model = ConditionNet(72, 1)
    #model = model.to('cuda')
    # 计算条件概率的对数 log p(y | x_t)

    #exit(1)
    #x_hat = x_hat.cpu()
    #y_condition = y_condition.cpu()


    log_p_y_given_x_t = model(x_hat, y_condition)


    # 计算损失（这里假设log_p_y_given_x_t就是我们要最大化的目标）
    # 例如，如果目标是让log p(y | x_t)最大化，我们可以取负值作为损失
    loss = -log_p_y_given_x_t.mean()

    # 反向传播，计算梯度
    loss.backward()


    # 获取 x_t 的梯度
    x_hat_grad = x_hat.grad
    #print(x_hat_grad)

    #exit(1)
    #print(d_cur)
    #print(x_hat_grad)
    #exit(1)
    para1 = 1
    para2 = 100000
    #当para1 = 1,para2 = 0时，为原版
    #当para1 <0 para2 = 0 时，往与训练数据相反的方向移动 当para1 > 0 para2 = 0往相同的方向移动
    #当para2 > 0时  ？
    #当para2 < 0时  ？
    x_next = x_hat + para1*((t_next - t_hat) * (d_cur - x_hat_grad * para2))

    #x_next = x_hat + (t_next - t_hat) * d_cur  #这里d_cur就是s

    # Apply 2nd order correction.
    if i < num_steps - 1:
        denoised = net(x_next, t_next).to(torch.float32)
        d_prime = (x_next - denoised) / t_next
        x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next

class VPLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t

    def __call__(self, denosie_fn, data, labels, augment_pipe=None):
        rnd_uniform = torch.rand([data.shape[0], 1, 1, 1], device=data.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(data) if augment_pipe is not None else (data, None)
        n = torch.randn_like(y) * sigma
        D_yn = denosie_fn(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()

#----------------------------------------------------------------------------
# Loss function corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

class VELoss:
    def __init__(self, sigma_min=0.02, sigma_max=100, D=128, N=3072, opts=None):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.D = D
        self.N = N
        print(f"In VE loss: D:{self.D}, N:{self.N}")

    def __call__(self, denosie_fn, data, labels = None, augment_pipe=None, stf=False, pfgmpp=False, ref_data=None):
        if pfgmpp:

            # N, 
            rnd_uniform = torch.rand(data.shape[0], device=data.device)
            sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)

            r = sigma.double() * np.sqrt(self.D).astype(np.float64)
            # Sampling form inverse-beta distribution
            samples_norm = np.random.beta(a=self.N / 2., b=self.D / 2.,
                                          size=data.shape[0]).astype(np.double)

            samples_norm = np.clip(samples_norm, 1e-3, 1-1e-3)

            inverse_beta = samples_norm / (1 - samples_norm + 1e-8)
            inverse_beta = torch.from_numpy(inverse_beta).to(data.device).double()
            # Sampling from p_r(R) by change-of-variable
            samples_norm = r * torch.sqrt(inverse_beta + 1e-8)
            samples_norm = samples_norm.view(len(samples_norm), -1)
            # Uniformly sample the angle direction
            gaussian = torch.randn(data.shape[0], self.N).to(samples_norm.device)
            unit_gaussian = gaussian / torch.norm(gaussian, p=2, dim=1, keepdim=True)
            # Construct the perturbation for x
            perturbation_x = unit_gaussian * samples_norm
            perturbation_x = perturbation_x.float()

            sigma = sigma.reshape((len(sigma), 1, 1, 1))
            weight = 1 / sigma ** 2
            y, augment_labels = augment_pipe(data) if augment_pipe is not None else (data, None)
            n = perturbation_x.view_as(y)
            D_yn = denosie_fn(y + n, sigma, labels,  augment_labels=augment_labels)
        else:
            rnd_uniform = torch.rand([data.shape[0], 1, 1, 1], device=data.device)
            sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
            weight = 1 / sigma ** 2
            y, augment_labels = augment_pipe(data) if augment_pipe is not None else (data, None)
            n = torch.randn_like(y) * sigma
            D_yn = denosie_fn(y + n, sigma, labels, augment_labels=augment_labels)

        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).

class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, hid_dim = 100, gamma=5, opts=None):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.hid_dim = hid_dim
        self.gamma = gamma
        self.opts = opts


    def __call__(self, denoise_fn, data):

        rnd_normal = torch.randn(data.shape[0], device=data.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()

        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

        y = data
        n = torch.randn_like(y) * sigma.unsqueeze(1)
        D_yn = denoise_fn(y + n, sigma)
    
        target = y
        loss = weight.unsqueeze(1) * ((D_yn - target) ** 2)

        return loss

