import argparse
import csv
import time
from statistics import mean
from types import SimpleNamespace
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# CHORDS Core (shape-agnostic)
# =========================
class CHORDS:
    def __init__(
        self,
        T: int,
        x0: torch.Tensor,
        num_cores: int,
        solver=lambda x_t, score_t, t, s: x_t + score_t * (s - t),
        init_t: Optional[str] = None,
        stopping_kwargs={
            "criterion": "core_index",
            "index": 1,
            "threshold": None,
        },
        verbose: bool = False,
    ):
        self.T = T
        K = T + 1
        self.x_ready = torch.ones(T + 1, K, dtype=torch.int64) * -1
        self.scores_ready = torch.ones(T + 1, K, dtype=torch.int64) * -1
        self.num_cores = num_cores
        print(f'[CHORDS] num_cores = {self.num_cores}')

        if init_t is None:
            raise NotImplementedError("Please provide initial points for CHORDS (e.g., '0-2-4').")
        init_t_list = [int(t) for t in init_t.split('-')]
        assert len(init_t_list) == num_cores, f"Expected {num_cores} initial points, got {len(init_t_list)}"
        assert all(0 <= t <= T for t in init_t_list), f"init_t values must be in [0, {T}]"
        assert all(init_t_list[i] < init_t_list[i+1] for i in range(len(init_t_list)-1)), "init_t must be strictly increasing"
        self.init_t = init_t_list
        print('[CHORDS] initial points =', self.init_t)

        self.solver = solver
        self.counter = 0
        self.stopping_kwargs = stopping_kwargs
        print('[CHORDS] stopping criteria =', self.stopping_kwargs)

        if stopping_kwargs["criterion"] not in ["core_index", "residual"]:
            raise ValueError("stopping_kwargs['criterion'] must be 'core_index' or 'residual'.")
        if stopping_kwargs["criterion"] == "core_index" and stopping_kwargs.get("index") is None:
            raise ValueError("stopping_kwargs['index'] must be provided for core_index criterion.")
        if stopping_kwargs["criterion"] == "residual" and stopping_kwargs.get("threshold") is None:
            raise ValueError("stopping_kwargs['threshold'] must be provided for residual criterion.")

        self.verbose = verbose

        # per-core state
        self.x_ready[0, 0] = self.counter
        self.cur_core_begin = [[0, 0, x0, None]]  # t, k, x, score
        self.cur_core_finish = [[None, None, None, None]]
        self.cur_core_status = [[0, 0, x0]]

        self.cur_core_to_compute = [0]  # queue of trajectories to score
        self.hits = []         # [(iter_idx, counter, x)]
        self.flops_count = 0   # number of score evaluations

    def get_allocation(self):
        cur_score_evals = []
        for core_id in self.cur_core_to_compute[:self.num_cores]:
            t, k, x = self.cur_core_status[core_id]
            assert self.scores_ready[t, k] == -1, f"[get_allocation] ({t},{k}) score already set"
            cur_score_evals.append((t, k, x))
        if self.verbose:
            print('*' * 10)
            print('cur_score_evals', [(t, k) for t, k, _ in cur_score_evals])
        return cur_score_evals

    def update_scores(self, scores):
        for compute_id, (t, k, score) in enumerate(scores):
            core_id = self.cur_core_to_compute[compute_id]
            assert (self.cur_core_status[core_id][0], self.cur_core_status[core_id][1]) == (t, k)
            self.cur_core_finish[core_id] = (t, k, self.cur_core_status[core_id][2], score)
            if (self.cur_core_begin[core_id][0], self.cur_core_begin[core_id][1]) == (t, k):
                assert self.cur_core_begin[core_id][3] is None
                self.cur_core_begin[core_id][3] = score
            self.scores_ready[t, k] = self.counter + 1
            self.flops_count += 1
        self.counter += 1

    def schedule_cores(self) -> Tuple[List[int], bool]:
        core_hit_idx = [core_id for core_id, (t, k, x) in enumerate(self.cur_core_status) if t == self.T]
        if len(core_hit_idx):
            print('[CHORDS] core_hit_idx =', core_hit_idx)
            for core_id in core_hit_idx:
                self.hits.append((self.cur_core_status[core_id][1], self.counter, self.cur_core_status[core_id][2]))
            self.cur_core_status = [core for core_id, core in enumerate(self.cur_core_status) if core_id not in core_hit_idx]
            self.cur_core_begin = [core for core_id, core in enumerate(self.cur_core_begin) if core_id not in core_hit_idx]
            self.cur_core_finish = [core for core_id, core in enumerate(self.cur_core_finish) if core_id not in core_hit_idx]

            if self.stopping_kwargs["criterion"] == "core_index":
                if len(self.hits) - 1 >= self.stopping_kwargs["index"]:
                    print(f'[CHORDS] stopping since core {len(self.hits) - 1} terminates')
                    return core_hit_idx, True
            elif self.stopping_kwargs["criterion"] == "residual":
                if len(self.hits) >= 2:
                    diff = torch.linalg.norm(self.hits[-1][-1] - self.hits[-2][-1]).double().item() / self.hits[-1][-1].numel()
                    if diff < self.stopping_kwargs["threshold"]:
                        print(f'[CHORDS] stopping: residual converged with diff {diff:.3e}')
                        return core_hit_idx, True

        if self.verbose:
            print('cur_core_begin', [(t, k) for t, k, x, score in self.cur_core_begin])
            print('cur_core_status', [(t, k) for t, k, x in self.cur_core_status])
            print('*' * 10)

        if len(self.cur_core_status) == 0:
            return [0], False
        return core_hit_idx, False

    def update_states(self, cnt: int):
        for core_id in self.cur_core_to_compute[:cnt]:
            core_to_init = None
            if core_id == len(self.cur_core_status) - 1:
                t_prev, k_prev, x_prev, score_prev = self.cur_core_finish[core_id]
                if k_prev == 0:
                    t_idx = self.init_t.index(t_prev)
                    if t_idx + 1 < len(self.init_t):
                        t_next = self.init_t[t_idx + 1]
                        x_next = self.solver(x_prev, score_prev, t_prev, t_next)
                        core_to_init = core_id + 1
                        self.cur_core_status.append([t_next, 0, x_next])
                        self.cur_core_begin.append([t_next, 0, x_next, None])
                        self.cur_core_finish.append([None, None, None, None])
                    else:
                        t_next = self.T
                        x_next = self.solver(x_prev, score_prev, t_prev, t_next)
                        self.hits.append((0, self.counter, x_next))

            hit_prev = core_id > 0 and self.cur_core_begin[core_id][0] == self.cur_core_finish[core_id - 1][0] and self.cur_core_begin[core_id][0] is not None
            if hit_prev:
                F = self.solver(self.cur_core_finish[core_id][2], self.cur_core_finish[core_id][3], self.cur_core_finish[core_id][0], self.cur_core_finish[core_id][0] + 1)
                G = self.solver(self.cur_core_begin[core_id][2],  self.cur_core_begin[core_id][3],  self.cur_core_begin[core_id][0],  self.cur_core_finish[core_id][0] + 1)
                cur_G = self.solver(self.cur_core_finish[core_id - 1][2], self.cur_core_finish[core_id - 1][3], self.cur_core_finish[core_id - 1][0], self.cur_core_finish[core_id][0] + 1)
                x = F - G + cur_G
                self.cur_core_begin[core_id] = [self.cur_core_status[core_id][0] + 1, self.cur_core_status[core_id][1] + 1, x, None]
                self.cur_core_status[core_id] = [self.cur_core_status[core_id][0] + 1, self.cur_core_status[core_id][1] + 1, x]
            else:
                F = self.solver(self.cur_core_finish[core_id][2], self.cur_core_finish[core_id][3], self.cur_core_finish[core_id][0], self.cur_core_finish[core_id][0] + 1)
                self.cur_core_status[core_id] = (self.cur_core_status[core_id][0] + 1, self.cur_core_status[core_id][1] + 1, F)

            self.cur_core_to_compute.append(core_id)
            if core_to_init is not None:
                self.cur_core_to_compute.append(core_to_init)

    def get_last_x_and_hittime(self):
        hit_iter_idx = [hit[0] for hit in self.hits]
        hit_time = [hit[1] for hit in self.hits]
        hit_x = [hit[2] for hit in self.hits]
        return hit_iter_idx, hit_x, hit_time

    def get_flops_count(self):
        return self.flops_count


# =========================
# Robust UNet loader: diffusers -> fallback TinyUNet
# =========================
def _locate_unet2dmodel():
    """
    Try multiple import paths for UNet2DModel.
    Return class if found; else return None (we'll fallback to TinyUNet2D).
    """
    try:
        from diffusers import UNet2DModel as _Cls
        return _Cls, "diffusers.UNet2DModel"
    except Exception:
        pass
    try:
        from diffusers.models.unets.unet_2d import UNet2DModel as _Cls
        return _Cls, "diffusers.models.unets.unet_2d.UNet2DModel"
    except Exception:
        pass
    try:
        from diffusers.models.unet_2d import UNet2DModel as _Cls
        return _Cls, "diffusers.models.unet_2d.UNet2DModel"
    except Exception:
        pass
    return None, None

import math

def _compute_rel_errors(x_serial: torch.Tensor, x_parallel: torch.Tensor):
    """返回 L2/L1/Linf 的相对误差（并把所有计算放到 cpu+float64，避免精度/设备差异）。"""
    xs = x_serial.detach().to("cpu", dtype=torch.float64).flatten()
    xp = x_parallel.detach().to("cpu", dtype=torch.float64).flatten()
    diff = xp - xs
    eps = 1e-12
    l2_rel  = diff.norm(p=2).item()   / (xs.norm(p=2).item()   + eps)
    l1_rel  = diff.abs().sum().item() / (xs.abs().sum().item() + eps)
    linf_rel = diff.abs().max().item() / (xs.abs().max().item() + eps)
    return {"rel_l2": l2_rel, "rel_l1": l1_rel, "rel_linf": linf_rel}

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # 跟随模块 dtype 的 buffer；model.to(dtype=...) 时会一起转换
        self.register_buffer("_dtype_token", torch.tensor(0.0), persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        dt = self._dtype_token.dtype
        dev = t.device
        t = t.to(dtype=dt).unsqueeze(1)  # (B,1)
        half = self.dim // 2
        base = torch.tensor(10000.0, device=dev, dtype=dt)
        freqs = torch.exp(torch.arange(half, device=dev, dtype=dt) * (-torch.log(base) / half))
        args = t * freqs  # (B, half)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if emb.shape[1] < self.dim:  # pad if odd
            emb = F.pad(emb, (0, self.dim - emb.shape[1]))
        # 明确用模块 dtype 输出
        return emb.to(dtype=dt)



class TinyUNet2D(nn.Module):
    """
    Very small UNet as a drop-in fallback.
    API: forward(x, t) -> SimpleNamespace(sample=out), where out has same shape as x.
    Time embedding is injected AFTER feature maps are projected to the matching channel size.
    """
    def __init__(self, in_channels: int, sample_size: int, base: int = 64):
        super().__init__()
        self.base = base

        # ---- time embedding (dtype-safe) ----
        class SinusoidalTimeEmbedding(nn.Module):
            def __init__(self, dim: int):
                super().__init__()
                self.dim = dim
                self.register_buffer("_dtype_token", torch.tensor(0.0), persistent=False)
            def forward(self, t: torch.Tensor) -> torch.Tensor:
                dt = self._dtype_token.dtype
                dev = t.device
                t = t.to(dtype=dt).unsqueeze(1)  # (B,1)
                half = self.dim // 2
                base = torch.tensor(10000.0, device=dev, dtype=dt)
                freqs = torch.exp(torch.arange(half, device=dev, dtype=dt) * (-torch.log(base) / half))
                args = t * freqs
                emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
                if emb.shape[1] < self.dim:
                    emb = F.pad(emb, (0, self.dim - emb.shape[1]))
                return emb.to(dtype=dt)

        # time MLP: tok -> fc_t -> SiLU
        self.tok  = SinusoidalTimeEmbedding(base * 2)
        self.fc_t = nn.Linear(base * 2, base * 2)
        self.act_t = nn.SiLU()

        ch1, ch2, ch3 = base, base * 2, base * 2

        # ---- encoder ----
        self.conv1a = nn.Conv2d(in_channels, ch1, 3, padding=1)
        self.act1a  = nn.SiLU()
        self.conv1b = nn.Conv2d(ch1, ch1, 3, padding=1)
        self.act1b  = nn.SiLU()

        self.down1  = nn.Conv2d(ch1, ch2, 4, stride=2, padding=1)
        self.conv2  = nn.Conv2d(ch2, ch2, 3, padding=1)
        self.act2   = nn.SiLU()

        self.down2  = nn.Conv2d(ch2, ch3, 4, stride=2, padding=1)

        # ---- middle ----
        self.mid_conv = nn.Conv2d(ch3, ch3, 3, padding=1)
        self.mid_act  = nn.SiLU()

        # ---- decoder ----
        self.up1      = nn.ConvTranspose2d(ch3, ch2, 4, stride=2, padding=1)
        self.dec1_conv= nn.Conv2d(ch2, ch2, 3, padding=1)
        self.dec1_act = nn.SiLU()

        self.up2       = nn.ConvTranspose2d(ch2, ch1, 4, stride=2, padding=1)
        self.dec2_conv1= nn.Conv2d(ch1, ch1, 3, padding=1)
        self.dec2_act1 = nn.SiLU()
        self.dec2_conv2= nn.Conv2d(ch1, in_channels, 3, padding=1)

        # time projections to match feature channels
        self.proj_t1 = nn.Linear(base * 2, ch1)
        self.proj_t2 = nn.Linear(base * 2, ch2)
        self.proj_t3 = nn.Linear(base * 2, ch3)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        B, C, H, W = x.shape

        # make sure dtype matches module weights dtype (handles float16/float32/bfloat16)
        w_dtype = self.fc_t.weight.dtype
        if x.dtype != w_dtype:
            x = x.to(dtype=w_dtype)

        # time embedding path
        temb = self.tok(t.reshape(B))
        temb = temb.to(dtype=self.fc_t.weight.dtype)
        temb = self.act_t(self.fc_t(temb))  # (B, 2*base)

        # ---- encoder with proper time injection (after channel projection) ----
        h1 = self.act1a(self.conv1a(x))                                    # (B, ch1, H, W)
        h1 = h1 + self.proj_t1(temb).view(B, -1, 1, 1)                     # add t in ch1
        h1 = self.act1b(self.conv1b(h1))                                   # (B, ch1, H, W)

        h2 = self.down1(h1)                                                # (B, ch2, H/2, W/2)
        h2 = self.act2(self.conv2(h2 + self.proj_t2(temb).view(B, -1, 1, 1)))  # inject at ch2

        h3 = self.down2(h2)                                                # (B, ch3, H/4, W/4)
        h3 = self.mid_act(self.mid_conv(h3 + self.proj_t3(temb).view(B, -1, 1, 1)))  # inject at ch3

        # ---- decoder ----
        u2 = self.up1(h3)                                                  # (B, ch2, H/2, W/2)
        u2 = self.dec1_act(self.dec1_conv(u2))

        u1 = self.up2(u2)                                                  # (B, ch1, H, W)
        u1 = self.dec2_act1(self.dec2_conv1(u1))
        out = self.dec2_conv2(u1)                                          # (B, C, H, W)

        out = torch.tanh(out)  # keep bounded; CHORDS uses Euler updates
        return SimpleNamespace(sample=out)



def build_unet_2d_model(in_channels: int, sample_size: int, base: int = 64, device="cpu", dtype=torch.float32):
    Cls, path = _locate_unet2dmodel()
    if Cls is not None:
        print(f"[UNet] Using {path}")
        try:
            unet = Cls(
                sample_size=sample_size,
                in_channels=in_channels,
                out_channels=in_channels,
                layers_per_block=2,
                block_out_channels=(base, base * 2, base * 2),
                down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"),
                up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D"),
            )
            return unet.to(device=device, dtype=dtype), "diffusers"
        except Exception as e:
            print(f"[UNet] Failed to instantiate diffusers UNet2DModel: {e}. Falling back to TinyUNet2D.")
    else:
        print("[UNet] diffusers UNet2DModel not found. Using TinyUNet2D fallback.")

    tiny = TinyUNet2D(in_channels=in_channels, sample_size=sample_size, base=base)
    return tiny.to(device=device, dtype=dtype), "tiny"


def make_unet_score_func(unet: nn.Module, T: int, score_scale: float = 1e-3):
    max_steps = 1000

    @torch.no_grad()
    def score_func(x: torch.Tensor, tau: int) -> torch.Tensor:
        if x.dim() == 3:  # (C,H,W) -> (1,C,H,W)
            x_in = x.unsqueeze(0)
        else:
            x_in = x
        B = x_in.shape[0]
        t_idx = int(round(float(tau) / max(T, 1) * (max_steps - 1)))
        t_tensor = torch.full((B,), t_idx, device=x_in.device, dtype=torch.long)
        out = unet(x_in, t_tensor).sample  # diffusers or tiny both return .sample
        return out * score_scale

    return score_func


# =========================
# Utils & runners
# =========================
def auto_init_t_linear(T: int, cores: int, leave_tail: int = 10) -> str:
    upper = max(0, T - leave_tail)
    pts = torch.linspace(0, upper, cores, dtype=torch.float64).tolist()
    uniq = []
    last = -1
    for v in pts:
        iv = int(round(v))
        if iv <= last:
            iv = last + 1
        iv = min(iv, upper)
        uniq.append(iv)
        last = iv
    return "-".join(str(v) for v in uniq)

def run_once(
    num_cores: int,
    init_t: str,
    T: int = 50,
    threshold: float = 1e-3,
    verbose: bool = False,
    base_seed: int = 10,
    # 模式与设备/精度
    use_unet: bool = False,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    dtype: str = "float32",  # "float32" | "float16" | "bfloat16"
    # UNet 形状参数（仅在 use_unet=True 时生效）
    batch: int = 1,
    in_channels: int = 4,
    height: int = 32,
    width: int = 32,
    score_scale: float = 1e-3,  # 缩放 UNet 输出，稳定 Euler 步
    # 合成 score 的维度（use_unet=False 时生效）
    D: int = 128,
    # 新增：是否返回最终解（最后一个 hit 的 x）
    return_last_x: bool = False,
):
    """
    单次 CHORDS 运行。若 return_last_x=True，返回 (metrics, last_x)，否则仅返回 metrics。
    metrics 字段包含：num_cores/init_t/hits/last_time/converge_time/flops/outer_iters/wall_time_s
    """
    import time
    torch.manual_seed(base_seed)

    # 解析 dtype
    if dtype == "float16":
        torch_dtype = torch.float16
    elif dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    # 选择 score 函数与初始状态 x0
    if use_unet:
        # 任意形状张量 (B,C,H,W)
        x0 = torch.randn(batch, in_channels, height, width, device=device, dtype=torch_dtype)
        # 需要你在文件里已定义：build_unet_2d_model / make_unet_score_func（含 dtype 安全处理）
        unet, backend = build_unet_2d_model(in_channels, sample_size=height, base=64, device=device, dtype=torch_dtype)
        if verbose:
            print(f"[UNet] backend = {backend}, x0.shape = {tuple(x0.shape)}, dtype = {x0.dtype}")
        score_func = make_unet_score_func(unet, T, score_scale=score_scale)
    else:
        # 合成线性 score（1D 向量）
        x0 = torch.randn(D, device=device, dtype=torch_dtype)
        score_func = lambda x, tau: 0.1 * x + 0.01 * tau

    # 构建并运行 CHORDS
    algo = CHORDS(
        T, x0, num_cores, init_t=init_t,
        stopping_kwargs={"criterion": "residual", "index": 1, "threshold": threshold},
        verbose=verbose,
    )

    outer_iters = 0
    t0 = time.perf_counter()
    while True:
        allocation = algo.get_allocation()
        if allocation == []:
            if verbose:
                print("[run_once] allocation empty, exit")
            break

        # 评估本批次 score
        scores = []
        for t, k, x in allocation:
            s = score_func(x, t)  # 已在 make_unet_score_func 中做 dtype 与形状安全处理
            scores.append((t, k, s))

        algo.update_scores(scores)
        algo.update_states(len(allocation))
        delete_ids, earlystop = algo.schedule_cores()
        outer_iters += 1

        if earlystop:
            if verbose:
                print(f"[run_once] early stop at outer-iter {outer_iters}")
            break

        # 出队已处理的任务
        algo.cur_core_to_compute = algo.cur_core_to_compute[len(allocation):]
        # 移除已命中的 core
        if len(delete_ids):
            algo.cur_core_to_compute = [cid for cid in algo.cur_core_to_compute if cid not in delete_ids]
    t1 = time.perf_counter()

    # 收集命中信息并估计“收敛时间”（相对最后一次 hit 的 sum(x) 差值 < 1）
    hit_iter_idx, hit_x, hit_time = algo.get_last_x_and_hittime()
    converge_iter = None
    if len(hit_x):
        last_sum = hit_x[-1].sum()
        for itr, x in enumerate(hit_x):
            if (x.sum() - last_sum).abs() < 1 and converge_iter is None:
                converge_iter = itr

    result = {
        "num_cores": num_cores,
        "init_t": init_t,
        "hits": len(hit_time),
        "last_time": hit_time[-1] if len(hit_time) else None,
        "converge_time": hit_time[converge_iter] if converge_iter is not None else None,
        "flops": algo.get_flops_count(),
        "outer_iters": outer_iters,
        "wall_time_s": t1 - t0,
    }

    if return_last_x:
        last_x = hit_x[-1] if len(hit_x) else None
        return result, last_x
    return result



def parse_cores_list(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip() != ""]


def pretty_print_results(rows: List[Dict]):
    header = ["cores", "FLOPs", "outer_iters", "last_time", "converge_time", "wall_time_s"]
    print("\n==== Scalability Summary ====")
    print("{:>7} | {:>10} | {:>11} | {:>9} | {:>13} | {:>11}".format(*header))
    print("-" * 72)
    for r in rows:
        print("{:>7} | {:>10} | {:>11} | {:>9} | {:>13} | {:>11.6f}".format(
            r["num_cores"],
            r["flops"],
            r["outer_iters"],
            str(r["last_time"]),
            str(r["converge_time"]),
            r["wall_time_s"],
        ))
    min_cores = min(r["num_cores"] for r in rows)
    base_time = next(r["wall_time_s"] for r in rows if r["num_cores"] == min_cores)
    print("\nSpeedup vs cores={} (by wall_time):".format(min_cores))
    for r in rows:
        sp = base_time / r["wall_time_s"] if r["wall_time_s"] > 0 else float('inf')
        print(f"  cores={r['num_cores']:<3d}  speedup={sp:.2f}x")


def export_csv(path: str, rows: List[Dict]):
    cols = ["num_cores", "init_t", "flops", "outer_iters", "last_time", "converge_time", "wall_time_s"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in cols})
    print(f"[CSV] Saved to {path}")


def maybe_plot(rows: List[Dict]):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] matplotlib not available: {e}")
        return
    rows = sorted(rows, key=lambda r: r["num_cores"])
    xs = [r["num_cores"] for r in rows]
    flops = [r["flops"] for r in rows]
    times = [r["wall_time_s"] for r in rows]

    plt.figure()
    plt.bar([str(x) for x in xs], flops)
    plt.title("CHORDS Scalability: FLOPs vs Cores")
    plt.xlabel("Cores")
    plt.ylabel("FLOPs (score evals)")
    plt.tight_layout()

    plt.figure()
    plt.bar([str(x) for x in xs], times)
    plt.title("CHORDS Scalability: Wall Time vs Cores")
    plt.xlabel("Cores")
    plt.ylabel("Wall Time (s)")
    plt.tight_layout()

    plt.show()
def relative_error_test(
    cores_list,                  # 并行核数列表，例如 [2,4]
    baseline_cores=1,            # 串行基线（固定为 1 即可）
    T=50,
    threshold=1e-3,
    repeats=1,                   # 多次重复，做均值
    leave_tail=10,
    verbose=False,
    # 下列参数与 run_once 一致，确保公平对比
    use_unet=False,
    device="cuda" if torch.cuda.is_available() else "cpu",
    dtype="float32",
    batch=1,
    in_channels=4,
    height=2048,
    width=2048,
    score_scale=1e-3,
    D=128,
):
    """
    返回列表：每个元素包含 cores、相对误差(rel_l2/rel_l1/rel_linf)、
    串行/并行平均 wall_time_s、FLOPs、以及 speedup 与 flops_ratio。
    """
    print("==== Relative Error: parallel vs serial (cores=1) ====")
    # 去掉 baseline 本身
    targets = [c for c in cores_list if c != baseline_cores]
    if not targets:
        print("[relerr] cores_list 为空或只有 baseline，略过。")
        return []

    results = []
    for c in targets:
        rels = []
        base_times, par_times = [], []
        base_flops, par_flops = [], []

        for r in range(repeats):
            seed = 1234 + r  # 保证对比一致的随机性

            # 自动生成 init_t
            init_t_base = auto_init_t_linear(T, baseline_cores, leave_tail=leave_tail)
            init_t_par  = auto_init_t_linear(T, c,              leave_tail=leave_tail)

            # 串行（baseline）
            base_metrics, x_serial = run_once(
                baseline_cores, init_t_base, T=T, threshold=threshold, verbose=verbose,
                base_seed=seed, use_unet=use_unet, device=device, dtype=dtype,
                batch=batch, in_channels=in_channels, height=height, width=width,
                score_scale=score_scale, D=D, return_last_x=True
            )
            if x_serial is None:
                print(f"[relerr] baseline 未产生 hit，跳过该次（seed={seed}）")
                continue

            # 并行（目标 cores）
            par_metrics, x_parallel = run_once(
                c, init_t_par, T=T, threshold=threshold, verbose=verbose,
                base_seed=seed, use_unet=use_unet, device=device, dtype=dtype,
                batch=batch, in_channels=in_channels, height=height, width=width,
                score_scale=score_scale, D=D, return_last_x=True
            )
            if x_parallel is None:
                print(f"[relerr] parallel cores={c} 未产生 hit,跳过该次（seed={seed}）")
                continue

            # 误差
            rel = _compute_rel_errors(x_serial, x_parallel)
            rels.append(rel)

            # 记录时间 & FLOPs
            base_times.append(base_metrics["wall_time_s"])
            par_times.append(par_metrics["wall_time_s"])
            base_flops.append(base_metrics["flops"])
            par_flops.append(par_metrics["flops"])

        # 聚合
        if len(rels) == 0:
            avg = {"rel_l2": float("nan"), "rel_l1": float("nan"), "rel_linf": float("nan")}
            t_base = t_par = speed = float("nan")
            f_base = f_par = fratio = float("nan")
        else:
            avg = {
                "rel_l2": sum(r["rel_l2"] for r in rels)/len(rels),
                "rel_l1": sum(r["rel_l1"] for r in rels)/len(rels),
                "rel_linf": sum(r["rel_linf"] for r in rels)/len(rels),
            }
            t_base = sum(base_times)/len(base_times)
            t_par  = sum(par_times)/len(par_times)
            speed  = (t_base / t_par) if t_par > 0 else float("inf")
            f_base = sum(base_flops)/len(base_flops)
            f_par  = sum(par_flops)/len(par_flops)
            fratio = (f_base / f_par) if f_par > 0 else float("inf")

        row = {
            "cores": c,
            **avg,
            "time_serial_s": t_base,
            "time_parallel_s": t_par,
            "speedup_time": speed,
            "flops_serial": f_base,
            "flops_parallel": f_par,
            "flops_ratio_serial_over_parallel": fratio,
        }
        results.append(row)

        print(f"[relerr] cores={c} | rel_l2={row['rel_l2']:.12e} "
              f"| rel_l1={row['rel_l1']:.12e} | rel_linf={row['rel_linf']:.12e} "
              f"| t_ser={row['time_serial_s']:.12f}s | t_par={row['time_parallel_s']:.6f}s "
              f"| speedup={row['speedup_time']:.12f}x")

    print("==== Done ====")
    return results

def export_relerr_csv(path: str, rows: list):
    cols = [
        "cores",
        "rel_l2", "rel_l1", "rel_linf",
        "time_serial_s", "time_parallel_s", "speedup_time",
        "flops_serial", "flops_parallel", "flops_ratio_serial_over_parallel",
    ]
    import csv
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    print(f"[CSV] Relative-error results saved to {path}")


def run_scalability(
    cores_list: List[int],
    T=50,
    threshold=1e-3,
    repeats: int = 1,
    leave_tail: int = 10,
    verbose=False,
    use_unet: bool = False,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    dtype: str = "float32",
    batch: int = 1,
    in_channels: int = 4,
    height: int = 32,
    width: int = 32,
    score_scale: float = 1e-3,
    D: int = 128,
) -> List[Dict]:
    print("==== Scalability Test ====")
    agg_rows: List[Dict] = []
    for cores in cores_list:
        init_t = auto_init_t_linear(T, cores, leave_tail=leave_tail)
        rep_rows = []
        for r in range(repeats):
            row = run_once(
                cores, init_t, T=T, threshold=threshold, verbose=verbose, base_seed=10 + r,
                use_unet=use_unet, device=device, dtype=dtype,
                batch=batch, in_channels=in_channels, height=height, width=width, score_scale=score_scale,
                D=D,
            )
            rep_rows.append(row)
            print(f"[cores={cores} rep={r+1}/{repeats}] init_t={init_t} | "
                  f"FLOPs={row['flops']} | wall_time={row['wall_time_s']:.6f}s | "
                  f"last_time={row['last_time']} | converge_time={row['converge_time']}")
        agg = {
            "num_cores": cores,
            "init_t": init_t,
            "flops": int(mean([r["flops"] for r in rep_rows])),
            "outer_iters": int(mean([r["outer_iters"] for r in rep_rows])),
            "last_time": int(mean([r["last_time"] for r in rep_rows if r["last_time"] is not None])) if any(r["last_time"] is not None for r in rep_rows) else None,
            "converge_time": int(mean([r["converge_time"] for r in rep_rows if r["converge_time"] is not None])) if any(r["converge_time"] is not None for r in rep_rows) else None,
            "wall_time_s": mean([r["wall_time_s"] for r in rep_rows]),
        }
        agg_rows.append(agg)
    print("==== Done ====")
    return agg_rows

def main():
    import argparse
    import torch

    parser = argparse.ArgumentParser(description="CHORDS scalability & relative-error tester")

    # 通用/任务参数
    parser.add_argument("--cores", type=str, default="1,2,4",
                        help="Comma-separated core counts, e.g., '1,2,4,8'")
    parser.add_argument("--T", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=1e-3,
                        help="Residual convergence threshold")
    parser.add_argument("--repeats", type=int, default=1,
                        help="Repeat each setting to average wall-time / relerr")
    parser.add_argument("--leave_tail", type=int, default=10,
                        help="Auto init_t spreads in [0, T - leave_tail]")
    parser.add_argument("--export_csv", type=str, default="",
                        help="(Scalability mode) export results to CSV")
    parser.add_argument("--plot", action="store_true",
                        help="(Scalability mode) show bar charts (needs matplotlib)")
    parser.add_argument("--verbose", action="store_true")

    # 相对误差测试
    parser.add_argument("--relerr", action="store_true",
                        help="Compare parallel vs serial (cores=1) relative errors")
    parser.add_argument("--relerr_csv", type=str, default="",
                        help="(Relative-error mode) export results to CSV")

    # UNet / 设备 / 精度
    parser.add_argument("--use_unet", action="store_true",
                        help="Use UNet forward as score function (diffusers or TinyUNet fallback)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "float16", "bfloat16"])

    # UNet 张量形状（仅在 --use_unet 时生效）
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--in_channels", type=int, default=4)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--score_scale", type=float, default=1e-3,
                        help="Scale UNet output to stabilize Euler step")

    # 合成 score（非 --use_unet 模式）
    parser.add_argument("--D", type=int, default=128)

    args = parser.parse_args()
    cores_list = parse_cores_list(args.cores)

    if args.relerr:
        # 并行 vs 串行相对误差（baseline 固定为 1）
        rel_rows = relative_error_test(
            cores_list=cores_list,
            baseline_cores=1,
            T=args.T,
            threshold=args.threshold,
            repeats=args.repeats,
            leave_tail=args.leave_tail,
            verbose=args.verbose,
            use_unet=args.use_unet,
            device=args.device,
            dtype=args.dtype,
            batch=args.batch,
            in_channels=args.in_channels,
            height=args.height,
            width=args.width,
            score_scale=args.score_scale,
            D=args.D,
        )

        print("\n==== Relative Error Summary (with time) ====")
        if len(rel_rows) == 0:
            print("(no results)")
        else:
            print("{:>7} | {:>10} | {:>10} | {:>10} | {:>12} | {:>12} | {:>9}".format(
                "cores", "rel_l2", "rel_l1", "rel_linf", "t_serial(s)", "t_parallel(s)", "speedup"
            ))
            print("-" * 88)
            for r in rel_rows:
                print("{:>7} | {:>10.3e} | {:>10.3e} | {:>10.3e} | {:>12.6f} | {:>12.6f} | {:>9.2f}".format(
                    r["cores"], r["rel_l2"], r["rel_l1"], r["rel_linf"],
                    r["time_serial_s"], r["time_parallel_s"], r["speedup_time"]
                ))

        if args.relerr_csv:
            export_relerr_csv(args.relerr_csv, rel_rows)
        return

    # 否则运行 scalability
    rows = run_scalability(
        cores_list=cores_list,
        T=args.T,
        threshold=args.threshold,
        repeats=args.repeats,
        leave_tail=args.leave_tail,
        verbose=args.verbose,
        use_unet=args.use_unet,
        device=args.device,
        dtype=args.dtype,
        batch=args.batch,
        in_channels=args.in_channels,
        height=args.height,
        width=args.width,
        score_scale=args.score_scale,
        D=args.D,
    )
    pretty_print_results(rows)
    if args.export_csv:
        export_csv(args.export_csv, rows)
    if args.plot:
        maybe_plot(rows)




if __name__ == "__main__":
    main()

"""
python benchmark.py --relerr --cores 1,2,4 \
  --use_unet --device cuda --dtype float16 --score_scale 1e-3 \
  --repeats 3 --relerr_csv relerr_unet.csv


python benchmark.py --relerr --cores 1,2,4 --relerr_csv relerr_results.csv

"""
