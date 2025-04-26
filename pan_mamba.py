

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn

from MMCNet.norm import LayerNorm, BiasFree_LayerNorm, WithBias_LayerNorm


try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.BasicConv2d = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, **kwargs),
            nn.BatchNorm2d(out_channels))

    def forward(self, x):
        x = F.relu(self.BasicConv2d(x))
        return x


class MSMamba(nn.Module):
    def __init__(self, use_reconstruct=False, gate_threshold=0.5, down_rate=1, dim_expand_rate=2, hidden_expand=4, patch_size=1, in_dim=64, in_size=16, zhilian=True):
        super().__init__()
        self.LN = LayerNorm(in_dim, None)
        self.down_rate = down_rate
        self.zhilian = zhilian
        self.res = nn.Upsample(scale_factor=1 / self.down_rate, mode='bilinear', align_corners=False)
        # self.res = nn.MaxPool2d(down_rate)

        self.dim_expand_rate = dim_expand_rate
        self.out_size = in_size // down_rate
        self.out_dim = in_dim * dim_expand_rate
        spa_hidden = in_size * in_size // patch_size // patch_size
        if spa_hidden < 256:
            spa_hidden = 256
        elif spa_hidden > 512:
            spa_hidden = 512
        self.spa_embed = SpaEmbedding(patch_size=patch_size, down_rate=down_rate, hidden=spa_hidden, in_size=in_size)
        self.spe_embed = SpeEmbedding(dim_expand_rate=dim_expand_rate, hidden=in_dim * hidden_expand, in_dim=in_dim)

        self.expand_spa = dim_expand_rate / patch_size / patch_size
        self.ssm_spa = SSM(d_model=in_dim * patch_size * patch_size, expand=self.expand_spa)

        self.expand_spe = 1 / down_rate / down_rate
        self.ssm_spe = SSMyz(d_model=in_size * in_size, expand=self.expand_spe)

        self.LN_out = WithBias_LayerNorm(self.out_dim)
        self.out_proj = nn.Linear(self.out_dim, in_dim, bias=False)
        self.use_reconstruct = use_reconstruct
        if self.use_reconstruct:
            self.sigomid = nn.Sigmoid()  # 创建 sigmoid 激活函数
            self.gate_treshold = gate_threshold  # 设置门控阈值
            self.gamma = self.LN_out.weight

    def forward(self, input):
        x = self.LN(input)
        spa = self.spa_embed(x)
        spe = self.spe_embed(x)
        outsize = self.out_size

        spe, z = self.ssm_spe(spe)
        z = z.permute(0, 2, 1)
        spa = self.ssm_spa(spa, z)

        out = self.LN_out(spa + spe.permute(0, 2, 1))

        if self.use_reconstruct:
            w_gamma = self.gamma / sum(self.gamma)  # 计算 gamma 权重
            reweights = self.sigomid(out * w_gamma)  # 计算重要性权重
            # print(torch.mean(reweights).item(), torch.max(reweights).item(), torch.min(reweights).item())
            # 门控机制
            info_mask = reweights >= self.gate_treshold  # 计算信息门控掩码
            noninfo_mask = reweights < self.gate_treshold  # 计算非信息门控掩码
            x_1 = info_mask * out  # 使用信息门控掩码
            x_2 = noninfo_mask * out  # 使用非信息门控掩码
            x_11, x_12 = torch.split(x_1, x_1.size(2) // 2, dim=2)  # 拆分特征为两部分
            x_21, x_22 = torch.split(x_2, x_2.size(2) // 2, dim=2)  # 拆分特征为两部分
            out = torch.cat([x_11 + x_22, x_12 + x_21], dim=2)
        out = self.out_proj(out)
        out = to_4d(out, h=outsize, w=outsize)

        if self.zhilian:
            if self.down_rate == 1:
                out += input
            else:
                out += self.res(input)

        return out


class PANMamba(nn.Module):
    def __init__(self, use_reconstruct=False, gate_threshold=0.5, down_rate=2, dim_expand_rate=2, hidden_expand=4, patch_size=2, in_dim=64, in_size=64, zhilian=True):
        super().__init__()
        self.down_rate = down_rate
        self.zhilian = zhilian
        self.res = nn.Upsample(scale_factor=1 / self.down_rate, mode='bilinear', align_corners=False)
        # self.res = nn.MaxPool2d(down_rate)
        self.dim_expand_rate = dim_expand_rate
        self.LN_in = LayerNorm(in_dim, None)
        spa_hidden = in_size * in_size // patch_size // patch_size
        if spa_hidden < 256:
            spa_hidden = 256
        elif spa_hidden > 512:
            spa_hidden = 512
        self.spa_embed = SpaEmbedding(patch_size=patch_size, down_rate=down_rate, hidden=spa_hidden, in_size=in_size)

        self.spe_embed = SpeEmbedding(dim_expand_rate=dim_expand_rate, hidden=in_dim * hidden_expand, in_dim=in_dim)
        self.out_dim = in_dim * dim_expand_rate
        self.out_size = in_size // down_rate
        self.expand_spa = dim_expand_rate / patch_size / patch_size
        self.ssm_spa = SSMyz(d_model=in_dim * patch_size * patch_size, expand=self.expand_spa)
        self.expand_spe = 1 / down_rate / down_rate
        self.ssm_spe = SSM(d_model=in_size * in_size, expand=self.expand_spe)
        self.LN_out = WithBias_LayerNorm(self.out_dim)
        self.out_proj = nn.Linear(self.out_dim, in_dim, bias=False)
        self.use_reconstruct = use_reconstruct
        if self.use_reconstruct:
            self.sigomid = nn.Sigmoid()  # 创建 sigmoid 激活函数
            self.gate_treshold = gate_threshold  # 设置门控阈值
            self.gamma = self.LN_out.weight
    def forward(self, input):
        x = self.LN_in(input)
        spa = self.spa_embed(x)
        spe = self.spe_embed(x)
        outsize = self.out_size
        spa, z = self.ssm_spa(spa)
        z = z.permute(0, 2, 1)
        spe = self.ssm_spe(spe, z)

        out = self.LN_out(spa + spe.permute(0, 2, 1))

        if self.use_reconstruct:
            w_gamma = self.gamma / sum(self.gamma)  # 计算 gamma 权重
            # w_gamma = w_gamma.unsqueeze(-1)
            reweights = self.sigomid(out * w_gamma)  # 计算重要性权重
            # print(w_gamma.size(), reweights.size(), out.size())
            # print(torch.mean(reweights).item(), torch.max(reweights).item(), torch.min(reweights).item())
            # 门控机制
            info_mask = reweights >= self.gate_treshold  # 计算信息门控掩码
            noninfo_mask = reweights < self.gate_treshold  # 计算非信息门控掩码
            x_1 = info_mask * out  # 使用信息门控掩码
            x_2 = noninfo_mask * out  # 使用非信息门控掩码
            x_11, x_12 = torch.split(x_1, x_1.size(2) // 2, dim=2)  # 拆分特征为两部分
            x_21, x_22 = torch.split(x_2, x_2.size(2) // 2, dim=2)  # 拆分特征为两部分
            out = torch.cat([x_11 + x_22, x_12 + x_21], dim=2)
        out = self.out_proj(out)

        out = to_4d(out, h=outsize, w=outsize)
        # print(input.size(), out.size())
        if self.zhilian:
            if self.down_rate == 1:
                out += input
            else:
                out += self.res(input)

        return out


class SpaEmbedding(nn.Module):
    def __init__(self, patch_size, down_rate, hidden, in_size):
        super().__init__()
        self.patch_size = patch_size
        self.unfold = nn.Unfold(kernel_size=(patch_size, patch_size), stride=patch_size)

        self.down_rate = down_rate

        self.channel = in_size * in_size // patch_size // patch_size

        self.conv = nn.Sequential(nn.Conv1d(self.channel, out_channels=hidden, kernel_size=1, stride=1, padding=0, bias=False),
                                  nn.SiLU(),
                                  nn.Conv1d(in_channels=hidden, out_channels=hidden, kernel_size=3, stride=1, padding=1),
                                  nn.SiLU(),
                                  nn.Conv1d(hidden, out_channels=in_size * in_size // down_rate // down_rate, kernel_size=1,
                                            stride=1, padding=0, bias=False),
                                  )
        # self.dim = in_dim * patch_size * patch_size
        # self.linear = nn.Sequential(nn.Linear(self.dim, hidden),
        #                             nn.SiLU(),
        #                             nn.Linear(hidden, hidden),
        #                             nn.SiLU(),
        #                             nn.Linear(hidden, in_dim * dim_expand_rate))

    def forward(self, x):
        batch, dim, h, w = x.shape

        out = self.unfold(x).permute(0, 2, 1)  # batch, h/ps * w/ps, c * ps * ps
        out = self.conv(out)
        # hw = h * w // self.down_rate // self.down_rate
        # out = self.linear(out.reshape(batch * hw, -1)).reshape(batch, hw, -1)

        return out


class SpeEmbedding(nn.Module):
    def __init__(self, dim_expand_rate, hidden, in_dim):
        super().__init__()
        self.dim_expand_rate = dim_expand_rate
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, out_channels=hidden, kernel_size=1, stride=1, padding=0, bias=False),
            nn.SiLU(),
            nn.Conv1d(in_channels=hidden, out_channels=hidden, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden, out_channels=in_dim * dim_expand_rate, kernel_size=1, stride=1, padding=0, bias=False),
        )
        # self.linear = nn.Sequential(nn.Linear(in_size * in_size, hidden),
        #                             nn.SiLU(),
        #                             nn.Linear(hidden, hidden),
        #                             nn.SiLU(),
        #                             nn.Linear(hidden, in_size * in_size // self.down_rate // self.down_rate))

    def forward(self, x):
        batch, dim, h, w = x.shape
        out = to_3d(x).permute(0, 2, 1)  # b, C, h*w
        out = self.conv(out)
        # out = self.linear(out.view(batch * dim * self.dim_expand_rate, -1)).view(batch, dim * self.dim_expand_rate, -1)

        return out


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class SSMyz(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        norm_type='BiasFree',
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        x, z = xz.chunk(2, dim=1)
        if self.in_proj.bias is not None:
            x = x + rearrange(self.in_proj.bias.to(dtype=x.dtype), "d -> d 1")

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # In the backward pass we write dx and dz next to each other to avoid torch.cat

        # Compute short convolution
        if conv_state is not None:
            # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
            # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
            conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)
        if causal_conv1d_fn is None:
            x = self.act(self.conv1d(x)[..., :seqlen])
        else:
            assert self.activation in ["silu", "swish"]
            x = causal_conv1d_fn(
                x=x,
                weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias,
                activation=self.activation,
            )

        # We're careful here about the layout, to avoid extra transposes.
        # We want dt to have d as the slowest moving dimension
        # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        assert self.activation in ["silu", "swish"]
        y = selective_scan_fn(
            x,
            dt,
            A,
            B,
            C,
            self.D.float(),
            z=z,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
            return_last_state=ssm_state is not None,
        )
        if ssm_state is not None:
            y, last_state = y
            ssm_state.copy_(last_state)
        out = rearrange(y, "b d l -> b l d")

        return out, z

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        return y.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.in_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state


class SSM(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        norm_type='BiasFree',
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.layer_idx = layer_idx
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

    def forward(self, hidden_states, z, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        # We do matmul and transpose BLH -> HBL at the same time
        x = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )

        if self.in_proj.bias is not None:
            x = x + rearrange(self.in_proj.bias.to(dtype=x.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, z, conv_state, ssm_state)
                return out

        # In the backward pass we write dx and dz next to each other to avoid torch.cat

        # Compute short convolution
        if conv_state is not None:
            # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
            # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
            conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)
        if causal_conv1d_fn is None:
            x = self.act(self.conv1d(x)[..., :seqlen])
        else:
            assert self.activation in ["silu", "swish"]
            x = causal_conv1d_fn(
                x=x,
                weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias,
                activation=self.activation,
            )

        # We're careful here about the layout, to avoid extra transposes.
        # We want dt to have d as the slowest moving dimension
        # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        assert self.activation in ["silu", "swish"]
        y = selective_scan_fn(
            x,
            dt,
            A,
            B,
            C,
            self.D.float(),
            z=z,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
            return_last_state=ssm_state is not None,
        )
        if ssm_state is not None:
            y, last_state = y
            ssm_state.copy_(last_state)
        out = rearrange(y, "b d l -> b l d")

        return out

    def step(self, hidden_states, z, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        x = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        z = z.squeeze(1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        return y.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state


class FusionMamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        use_sobel=False,
        use_exchange=False,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.layer_idx = layer_idx
        self.use_exchange = use_exchange
        self.use_sobel = use_sobel
        self.sobel = Sobelxy()

        self.norm_pan = WithBias_LayerNorm(self.d_model)
        self.norm_ms = WithBias_LayerNorm(self.d_model)

        self.in_proj_pan = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.in_proj_ms = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.in_proj_z_pan = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.in_proj_z_ms = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)

        self.conv1d_pan = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        self.conv1d_ms = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj_pan = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)
        self.x_proj_ms = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)
        self.dt_proj_pan = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        self.dt_proj_ms = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj_pan.weight, dt_init_std)
            nn.init.constant_(self.dt_proj_ms.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj_pan.weight, -dt_init_std, dt_init_std)
            nn.init.uniform_(self.dt_proj_ms.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj_pan.bias.copy_(inv_dt)
            self.dt_proj_ms.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj_pan.bias._no_reinit = True
        self.dt_proj_ms.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D_pan = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D_ms = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D_pan._no_weight_decay = True
        self.D_ms._no_weight_decay = True

        self.out_proj_pan = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.out_proj_ms = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, pan, ms):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        h_pan, w_pan = pan.shape[2:]
        h_ms, w_ms = ms.shape[2:]
        sobel_pan = self.sobel(pan)  # b, 1, h, w

        if self.use_exchange:
            x_11, x_12 = torch.split(pan, pan.size(1) // 2, dim=1)  # 拆分特征为两部分
            x_21, x_22 = torch.split(ms, ms.size(1) // 2, dim=1)  # 拆分特征为两部分
            pan = torch.cat([x_11, x_22], dim=1) + pan
            ms = torch.cat([x_21, x_12], dim=1) + ms

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        pan = self.norm_pan(to_3d(pan))
        ms = self.norm_ms(to_3d(ms))
        batch, seqlen_pan, dim = pan.shape
        batch, seqlen_ms, dim = ms.shape

        # We do matmul and transpose BLH -> HBL at the same time

        x_pan = rearrange(
            self.in_proj_pan.weight @ rearrange(pan, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen_pan,
        )
        x_ms = rearrange(
            self.in_proj_ms.weight @ rearrange(ms, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen_ms,
        )
        if self.in_proj_pan.bias is not None:
            x_pan = x_pan + rearrange(self.in_proj_pan.bias.to(dtype=x_pan.dtype), "d -> d 1")
        if self.in_proj_ms.bias is not None:
            x_ms = x_ms + rearrange(self.in_proj_ms.bias.to(dtype=x_ms.dtype), "d -> d 1")

        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if self.use_sobel:
            z_pan = to_3d(sobel_pan) * pan
            # print(sobel_pan.size(), pan.size())
            z_ms = to_3d(1 - sobel_pan) * ms
            if self.expand == 1:
                z_pan = z_pan.permute(0, 2, 1)
                z_ms = z_ms.permute(0, 2, 1)
            else:
                z_pan = rearrange(
                    self.in_proj_z_pan.weight @ rearrange(z_pan, "b l d -> d (b l)"),
                    "d (b l) -> b d l",
                    l=seqlen_pan,
                )
                z_ms = rearrange(
                    self.in_proj_z_ms.weight @ rearrange(z_ms, "b l d -> d (b l)"),
                    "d (b l) -> b d l",
                    l=seqlen_ms,
                )
                if self.in_proj_z_pan.bias is not None:
                    z_pan = z_pan + rearrange(self.in_proj_z_pan.bias.to(dtype=z_pan.dtype), "d -> d 1")
                if self.in_proj_ms.bias is not None:
                    z_ms = z_ms + rearrange(self.in_proj_z_ms.bias.to(dtype=z_ms.dtype), "d -> d 1")

        else:
            z_pan = rearrange(
                self.in_proj_z_pan.weight @ rearrange(pan, "b l d -> d (b l)"),
                "d (b l) -> b d l",
                l=seqlen_pan,
            )
            z_ms = rearrange(
                self.in_proj_z_ms.weight @ rearrange(ms, "b l d -> d (b l)"),
                "d (b l) -> b d l",
                l=seqlen_ms,
            )
            if self.in_proj_z_pan.bias is not None:
                z_pan = z_pan + rearrange(self.in_proj_z_pan.bias.to(dtype=z_pan.dtype), "d -> d 1")
            if self.in_proj_ms.bias is not None:
                z_ms = z_ms + rearrange(self.in_proj_z_ms.bias.to(dtype=z_ms.dtype), "d -> d 1")

        assert self.activation in ["silu", "swish"]
        x_pan = causal_conv1d_fn(
            x=x_pan,
            weight=rearrange(self.conv1d_pan.weight, "d 1 w -> d w"),
            bias=self.conv1d_pan.bias,
            activation=self.activation,
        )
        x_ms = causal_conv1d_fn(
            x=x_ms,
            weight=rearrange(self.conv1d_ms.weight, "d 1 w -> d w"),
            bias=self.conv1d_ms.bias,
            activation=self.activation,
        )

        # We're careful here about the layout, to avoid extra transposes.
        # We want dt to have d as the slowest moving dimension
        # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
        x_dbl_pan = self.x_proj_pan(rearrange(x_pan, "b d l -> (b l) d"))  # (bl d)
        dt_pan, B_pan, C_pan = torch.split(x_dbl_pan, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt_pan = self.dt_proj_pan.weight @ dt_pan.t()
        dt_pan = rearrange(dt_pan, "d (b l) -> b d l", l=seqlen_pan)
        B_pan = rearrange(B_pan, "(b l) dstate -> b dstate l", l=seqlen_pan).contiguous()
        C_pan = rearrange(C_pan, "(b l) dstate -> b dstate l", l=seqlen_pan).contiguous()

        x_dbl_ms = self.x_proj_ms(rearrange(x_ms, "b d l -> (b l) d"))  # (bl d)
        dt_ms, B_ms, C_ms = torch.split(x_dbl_ms, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt_ms = self.dt_proj_ms.weight @ dt_ms.t()
        dt_ms = rearrange(dt_ms, "d (b l) -> b d l", l=seqlen_ms)
        B_ms = rearrange(B_ms, "(b l) dstate -> b dstate l", l=seqlen_ms).contiguous()
        C_ms = rearrange(C_ms, "(b l) dstate -> b dstate l", l=seqlen_ms).contiguous()

        assert self.activation in ["silu", "swish"]
        y_pan = selective_scan_fn(
            x_pan,
            dt_pan,
            A,
            B_pan,
            C_pan,
            self.D_pan.float(),
            z=z_pan,
            delta_bias=self.dt_proj_pan.bias.float(),
            delta_softplus=True,
            return_last_state=None,
        )
        y_ms = selective_scan_fn(
            x_ms,
            dt_ms,
            A,
            B_ms,
            C_ms,
            self.D_ms.float(),
            z=z_ms,
            delta_bias=self.dt_proj_ms.bias.float(),
            delta_softplus=True,
            return_last_state=None,
        )
        y_pan = rearrange(y_pan, "b d l -> b l d")
        y_ms = rearrange(y_ms, "b d l -> b l d")
        pan = self.out_proj_pan(y_pan)
        ms = self.out_proj_ms(y_ms)

        pan = to_4d(pan, h=h_pan, w=w_pan)
        ms = to_4d(ms, h=h_ms, w=w_ms)

        return pan, ms


class Exchange(nn.Module):
    def __init__(self, exchange_ratio=0.5, expand=4, learn=False):
        super(Exchange, self).__init__()
        self.expand = expand
        if learn:
            self.exchange_ratio = nn.Parameter(torch.tensor(exchange_ratio))
        else:
            self.exchange_ratio = exchange_ratio

    def forward(self, x1, x2, bn1, bn2):  # batch, C, hw ; C
        bn1, bn2 = bn1.abs(), bn2.abs()

        topk = int(bn1.size(0) * (1. - self.exchange_ratio))
        _, indice1 = torch.topk(bn1, topk)
        id1 = torch.full_like(bn1, False, dtype=bool)
        id1[indice1] = True
        _, indice2 = torch.topk(bn2, topk)
        id2 = torch.full_like(bn2, False, dtype=bool)
        id2[indice2] = True

        y1, y2 = torch.zeros_like(x1), torch.zeros_like(x2)
        y1[:, id1, :] = x1[:, id1, :]
        y1[:, ~id1, :] = x2[:, ~id2, :]
        y2[:, id2, :] = x2[:, id2, :]
        y2[:, ~id2, :] = x1[:, ~id1, :]
        return y1, y2


class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False)
        self.weighty = nn.Parameter(data=kernely, requires_grad=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.mean(x, dim=1, keepdim=True)

        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return self.sigmoid(torch.abs(sobelx)+torch.abs(sobely))


