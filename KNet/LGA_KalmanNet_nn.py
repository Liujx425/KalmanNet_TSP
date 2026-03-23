"""
LGA-KalmanNet: Local Geometry Attention Enhanced Kalman Neural Network

Integrates LGA's adaptive distance metric learning into KalmanNet's
Kalman gain computation for robust single-target tracking.

Architecture:
    1. Observation Buffer: Sliding window of recent observations
    2. LGA Attention: Adaptive metric-based attention over observation buffer
       to compute robust observation features (down-weights outliers)
    3. KalmanNet Core: GRU-based Kalman gain estimation enhanced with
       LGA-processed features
    4. State Update: Standard Kalman predict-update with learned gain
"""

import torch
import torch.nn as nn
import torch.nn.functional as func


class LGABlock(nn.Module):
    """
    Local Geometry Attention block for observation robustification.

    Processes a buffer of recent observations using LGA attention to
    produce a robust observation representation that down-weights outliers.
    """
    def __init__(self, obs_dim, d_model, n_heads=2, d_G=32, eps=1e-2, dropout=0.0):
        super().__init__()
        self.obs_dim = obs_dim
        self.d_model = d_model
        self.n_heads = n_heads
        assert d_model % n_heads == 0
        self.d_h = d_model // n_heads
        self.eps = eps
        self.scale = self.d_h ** -0.5

        # Project observations to d_model
        self.obs_embed = nn.Linear(obs_dim, d_model)

        # Q, K, V projections
        self.W_Q = nn.Linear(d_model, d_model, bias=True)
        self.W_K = nn.Linear(d_model, d_model, bias=True)
        self.W_V = nn.Linear(d_model, d_model, bias=True)

        # Learnable metric tensor networks (one per head)
        self.G_qs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.d_h, d_G),
                nn.GELU(),
                nn.Linear(d_G, self.d_h),
                nn.Softplus()
            ) for _ in range(n_heads)
        ])

        # Output projection
        self.to_out = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout)
        )

        # Layer norm
        self.norm = nn.LayerNorm(d_model)

        self.G_loss = 0.0

    def compute_metric_tensor(self, q, k):
        """Estimate adaptive distance metrics via sampling."""
        bs, q_len, n_heads, d_h = q.shape
        num_samples = min(64, bs) * 2
        if num_samples > bs:
            num_samples = bs
        sample_indices = torch.randperm(bs, device=q.device)[:num_samples]

        q_samples = torch.cat((
            q[sample_indices],
            torch.empty(num_samples, q_len, n_heads, d_h, device=q.device).uniform_(-3, 3)
        ), dim=1)
        k_samples = k[sample_indices]

        k_sq = torch.einsum('bsnd,bsnd->bsn', k_samples, k_samples).unsqueeze(1)
        qk = torch.einsum('bpnd,bsnd->bpsn', q_samples, k_samples)
        sq_dist = (k_sq - 2 * qk) * self.scale
        weights = F.softmax(-sq_dist, dim=2)

        q_expanded = q_samples.unsqueeze(2)
        k_expanded = k_samples.unsqueeze(1)
        squared_diff = (q_expanded - k_expanded) ** 2
        weighted_squared_diff = weights.unsqueeze(-1) * squared_diff
        Sigma_diag = torch.sum(weighted_squared_diff, dim=2)
        G_samples = torch.reciprocal(Sigma_diag + self.eps)

        self.G_loss = 0.0
        for i in range(self.n_heads):
            self.G_loss += func.mse_loss(
                G_samples[:, :, i, :],
                self.G_qs[i](q_samples[:, :, i, :])
            )
        self.G_loss /= self.n_heads

    def compute_mahalanobis_attention(self, q, k, G):
        """Compute attention scores using Mahalanobis distance."""
        k_sq = k * k
        q_G_k = torch.einsum('bpnd,bsnd->bpns', q * G, k)
        k_G_k = torch.einsum('bpnd,bsnd->bpns', G, k_sq)
        return 2 * q_G_k - k_G_k

    def forward(self, obs_buffer):
        """
        Args:
            obs_buffer: [batch_size, window_size, obs_dim]
        Returns:
            robust_obs: [batch_size, d_model] - robust observation feature
            G_loss: metric learning loss (training only)
        """
        bs, win_size, _ = obs_buffer.shape

        # Embed observations
        x = self.obs_embed(obs_buffer)  # [bs, win_size, d_model]
        x = self.norm(x)

        # Use the latest observation as query, all as keys/values
        q_s = self.W_Q(x).view(bs, win_size, self.n_heads, self.d_h)
        k_s = self.W_K(x).view(bs, win_size, self.n_heads, self.d_h)
        v_s = self.W_V(x).view(bs, win_size, self.n_heads, self.d_h)

        # Compute learned metric tensors
        Gs = [self.G_qs[i](q_s[:, :, i, :]) for i in range(self.n_heads)]
        G = torch.stack(Gs, dim=2)

        # Compute metric tensor loss during training
        if self.training and bs >= 2:
            self.compute_metric_tensor(q_s.detach(), k_s.detach())

        # Compute Mahalanobis attention
        attn_scores = self.compute_mahalanobis_attention(q_s, k_s, G.detach())
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Compute output
        output = torch.einsum('bpns,bsnd->bpnd', attn_weights, v_s)
        output = output.contiguous().view(bs, win_size, self.n_heads * self.d_h)
        output = self.to_out(output)

        # Take the last position (most recent) as the robust representation
        robust_obs = output[:, -1, :]  # [bs, d_model]

        return robust_obs, self.G_loss


class LGA_KalmanNetNN(nn.Module):
    """
    LGA-Enhanced KalmanNet for Robust Single Target Tracking.

    Combines:
    - LGA attention module for robust observation processing
    - KalmanNet's GRU-based Kalman gain estimation
    - Standard Kalman predict-update framework
    """

    def __init__(self):
        super().__init__()

    def NNBuild(self, SysModel, args):
        """Build the LGA-KalmanNet architecture."""
        if args.use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.InitSystemDynamics(SysModel.f, SysModel.h, SysModel.m, SysModel.n)

        # LGA parameters
        self.window_size = getattr(args, 'lga_window', 5)
        self.d_model_lga = getattr(args, 'lga_d_model', 16)
        self.n_heads_lga = getattr(args, 'lga_n_heads', 2)
        self.d_G = getattr(args, 'lga_d_G', 32)
        self.lga_eps = getattr(args, 'lga_eps', 1e-2)
        self.lga_weight = getattr(args, 'lga_weight', 0.1)

        # Build LGA block
        self.lga_block = LGABlock(
            obs_dim=self.n,
            d_model=self.d_model_lga,
            n_heads=self.n_heads_lga,
            d_G=self.d_G,
            eps=self.lga_eps
        ).to(self.device)

        # Project LGA features to observation-like space for fusion
        self.lga_proj = nn.Sequential(
            nn.Linear(self.d_model_lga, self.n),
            nn.Tanh()
        ).to(self.device)

        # Build KalmanNet gain network
        self.InitKGainNet(SysModel.prior_Q, SysModel.prior_Sigma, SysModel.prior_S, args)

    def InitKGainNet(self, prior_Q, prior_Sigma, prior_S, args):
        """Initialize the Kalman Gain estimation network."""
        self.seq_len_input = 1
        self.batch_size = args.n_batch

        self.prior_Q = prior_Q.to(self.device)
        self.prior_Sigma = prior_Sigma.to(self.device)
        self.prior_S = prior_S.to(self.device)

        in_mult = args.in_mult_KNet
        out_mult = args.out_mult_KNet

        # GRU to track Q
        self.d_input_Q = self.m * in_mult
        self.d_hidden_Q = self.m ** 2
        self.GRU_Q = nn.GRU(self.d_input_Q, self.d_hidden_Q).to(self.device)

        # GRU to track Sigma
        self.d_input_Sigma = self.d_hidden_Q + self.m * in_mult
        self.d_hidden_Sigma = self.m ** 2
        self.GRU_Sigma = nn.GRU(self.d_input_Sigma, self.d_hidden_Sigma).to(self.device)

        # GRU to track S (enhanced with LGA features)
        # Extra input from LGA robust features
        self.d_lga_feature = self.n  # projected LGA feature dim
        self.d_input_S = self.n ** 2 + 2 * self.n * in_mult + self.d_lga_feature
        self.d_hidden_S = self.n ** 2
        self.GRU_S = nn.GRU(self.d_input_S, self.d_hidden_S).to(self.device)

        # FC1: Sigma -> S input
        self.d_input_FC1 = self.d_hidden_Sigma
        self.d_output_FC1 = self.n ** 2
        self.FC1 = nn.Sequential(
            nn.Linear(self.d_input_FC1, self.d_output_FC1),
            nn.ReLU()
        ).to(self.device)

        # FC2: Compute Kalman Gain
        self.d_input_FC2 = self.d_hidden_S + self.d_hidden_Sigma
        self.d_output_FC2 = self.n * self.m
        self.d_hidden_FC2 = self.d_input_FC2 * out_mult
        self.FC2 = nn.Sequential(
            nn.Linear(self.d_input_FC2, self.d_hidden_FC2),
            nn.ReLU(),
            nn.Linear(self.d_hidden_FC2, self.d_output_FC2)
        ).to(self.device)

        # FC3: Backward flow
        self.d_input_FC3 = self.d_hidden_S + self.d_output_FC2
        self.d_output_FC3 = self.m ** 2
        self.FC3 = nn.Sequential(
            nn.Linear(self.d_input_FC3, self.d_output_FC3),
            nn.ReLU()
        ).to(self.device)

        # FC4: Backward flow
        self.d_input_FC4 = self.d_hidden_Sigma + self.d_output_FC3
        self.d_output_FC4 = self.d_hidden_Sigma
        self.FC4 = nn.Sequential(
            nn.Linear(self.d_input_FC4, self.d_output_FC4),
            nn.ReLU()
        ).to(self.device)

        # FC5: Forward update diff
        self.d_input_FC5 = self.m
        self.d_output_FC5 = self.m * in_mult
        self.FC5 = nn.Sequential(
            nn.Linear(self.d_input_FC5, self.d_output_FC5),
            nn.ReLU()
        ).to(self.device)

        # FC6: Forward evolution diff
        self.d_input_FC6 = self.m
        self.d_output_FC6 = self.m * in_mult
        self.FC6 = nn.Sequential(
            nn.Linear(self.d_input_FC6, self.d_output_FC6),
            nn.ReLU()
        ).to(self.device)

        # FC7: Observation diffs
        self.d_input_FC7 = 2 * self.n
        self.d_output_FC7 = 2 * self.n * in_mult
        self.FC7 = nn.Sequential(
            nn.Linear(self.d_input_FC7, self.d_output_FC7),
            nn.ReLU()
        ).to(self.device)

    def InitSystemDynamics(self, f, h, m, n):
        self.f = f
        self.m = m
        self.h = h
        self.n = n

    def InitSequence(self, M1_0, T):
        """
        Initialize sequence.
        Args:
            M1_0: initial state [batch_size, m, 1]
            T: sequence length
        """
        self.T = T
        self.m1x_posterior = M1_0.to(self.device)
        self.m1x_posterior_previous = self.m1x_posterior
        self.m1x_prior_previous = self.m1x_posterior
        self.y_previous = self.h(self.m1x_posterior)

        # Initialize observation buffer with predicted observations
        y0 = self.y_previous.squeeze(2)  # [bs, n]
        self.obs_buffer = y0.unsqueeze(1).repeat(1, self.window_size, 1)  # [bs, W, n]

    def update_obs_buffer(self, y):
        """Shift observation buffer and append new observation."""
        y_squeezed = y.squeeze(2)  # [bs, n]
        self.obs_buffer = torch.cat([
            self.obs_buffer[:, 1:, :],
            y_squeezed.unsqueeze(1)
        ], dim=1)

    def step_prior(self):
        """Predict the next state."""
        self.m1x_prior = self.f(self.m1x_posterior)
        self.m1y = self.h(self.m1x_prior)

    def step_KGain_est(self, y):
        """Estimate Kalman Gain using KalmanNet + LGA features."""
        # Compute difference features
        obs_diff = torch.squeeze(y, 2) - torch.squeeze(self.y_previous, 2)
        obs_innov_diff = torch.squeeze(y, 2) - torch.squeeze(self.m1y, 2)
        fw_evol_diff = torch.squeeze(self.m1x_posterior, 2) - torch.squeeze(self.m1x_posterior_previous, 2)
        fw_update_diff = torch.squeeze(self.m1x_posterior, 2) - torch.squeeze(self.m1x_prior_previous, 2)

        # Normalize
        obs_diff = func.normalize(obs_diff, p=2, dim=1, eps=1e-12)
        obs_innov_diff = func.normalize(obs_innov_diff, p=2, dim=1, eps=1e-12)
        fw_evol_diff = func.normalize(fw_evol_diff, p=2, dim=1, eps=1e-12)
        fw_update_diff = func.normalize(fw_update_diff, p=2, dim=1, eps=1e-12)

        # LGA: process observation buffer for robust features
        robust_feat, self.current_G_loss = self.lga_block(self.obs_buffer)
        lga_feature = self.lga_proj(robust_feat)  # [bs, n]
        lga_feature = func.normalize(lga_feature, p=2, dim=1, eps=1e-12)

        # Compute Kalman Gain
        KG = self.KGain_step(obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff, lga_feature)

        self.KGain = torch.reshape(KG, (self.batch_size, self.m, self.n))

    def KNet_step(self, y):
        """Main forward step: predict, estimate gain, update."""
        # Predict
        self.step_prior()

        # Update observation buffer
        self.update_obs_buffer(y)

        # Estimate Kalman Gain
        self.step_KGain_est(y)

        # Innovation
        dy = y - self.m1y  # [batch_size, n, 1]

        # Update state
        INOV = torch.bmm(self.KGain, dy)
        self.m1x_posterior_previous = self.m1x_posterior
        self.m1x_posterior = self.m1x_prior + INOV

        self.m1x_prior_previous = self.m1x_prior
        self.y_previous = y

        return self.m1x_posterior

    def KGain_step(self, obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff, lga_feature):
        """Compute Kalman Gain with LGA-enhanced S-GRU."""
        def expand_dim(x):
            expanded = torch.empty(self.seq_len_input, self.batch_size, x.shape[-1]).to(self.device)
            expanded[0, :, :] = x
            return expanded

        obs_diff = expand_dim(obs_diff)
        obs_innov_diff = expand_dim(obs_innov_diff)
        fw_evol_diff = expand_dim(fw_evol_diff)
        fw_update_diff = expand_dim(fw_update_diff)
        lga_feature = expand_dim(lga_feature)

        # Forward Flow
        # FC5
        out_FC5 = self.FC5(fw_update_diff)

        # Q-GRU
        out_Q, self.h_Q = self.GRU_Q(out_FC5, self.h_Q)

        # FC6
        out_FC6 = self.FC6(fw_evol_diff)

        # Sigma-GRU
        in_Sigma = torch.cat((out_Q, out_FC6), 2)
        out_Sigma, self.h_Sigma = self.GRU_Sigma(in_Sigma, self.h_Sigma)

        # FC1
        out_FC1 = self.FC1(out_Sigma)

        # FC7
        in_FC7 = torch.cat((obs_diff, obs_innov_diff), 2)
        out_FC7 = self.FC7(in_FC7)

        # S-GRU (enhanced with LGA features)
        in_S = torch.cat((out_FC1, out_FC7, lga_feature), 2)
        out_S, self.h_S = self.GRU_S(in_S, self.h_S)

        # FC2: Kalman Gain
        in_FC2 = torch.cat((out_Sigma, out_S), 2)
        out_FC2 = self.FC2(in_FC2)

        # Backward Flow
        # FC3
        in_FC3 = torch.cat((out_S, out_FC2), 2)
        out_FC3 = self.FC3(in_FC3)

        # FC4
        in_FC4 = torch.cat((out_Sigma, out_FC3), 2)
        out_FC4 = self.FC4(in_FC4)

        # Update Sigma hidden state
        self.h_Sigma = out_FC4

        return out_FC2

    def forward(self, y):
        y = y.to(self.device)
        return self.KNet_step(y)

    def init_hidden_KNet(self):
        """Initialize GRU hidden states."""
        weight = next(self.parameters()).data
        hidden = weight.new(self.seq_len_input, self.batch_size, self.d_hidden_S).zero_()
        self.h_S = hidden.data
        self.h_S = self.prior_S.flatten().reshape(1, 1, -1).repeat(
            self.seq_len_input, self.batch_size, 1)

        hidden = weight.new(self.seq_len_input, self.batch_size, self.d_hidden_Sigma).zero_()
        self.h_Sigma = hidden.data
        self.h_Sigma = self.prior_Sigma.flatten().reshape(1, 1, -1).repeat(
            self.seq_len_input, self.batch_size, 1)

        hidden = weight.new(self.seq_len_input, self.batch_size, self.d_hidden_Q).zero_()
        self.h_Q = hidden.data
        self.h_Q = self.prior_Q.flatten().reshape(1, 1, -1).repeat(
            self.seq_len_input, self.batch_size, 1)

        # Reset G_loss
        self.current_G_loss = 0.0

    def get_G_loss(self):
        """Return the current LGA metric learning loss."""
        return self.current_G_loss
