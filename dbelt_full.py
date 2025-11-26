# ---------------------------------------------------------------------------
# 8.17duetl_full.py  ——  DUET-L (CIFAR)  •  PyTorch ≥ 2.1
#
# 运行示例：#D:\appp\DUELT\duel-env\Scripts\Activate.ps1
#   python D:\appp\DUELT\CIFAR\duetl_cifar10_lt_opt.py  --datapath D:\appp\DUELT\CIFAR\data --lt_dir  D:\appp\DUELT\CIFAR\data\cifar-10-LT-10 --epochs 300 --batch_size 256 --lr 0.1 --num_classes 10 --K 4 --N_bar 2 --lambda_rs 0.3 --lambda_bt 0.06 --use_load_balance true --lambda_M 0.3 --amp false --gpu 0 --seed 42


import os, argparse, tqdm, numpy as np, time
from collections import defaultdict
from typing import Dict, Tuple
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from pathlib import Path
import math, pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, Sampler
from torchvision import datasets, transforms
try:  import yaml
except ImportError:  yaml = None
try:
    autocast = torch.amp.autocast         
except AttributeError:                    
    from torch.cuda.amp import autocast
try:
    from tensorboardX import SummaryWriter
except Exception:
    from torch.utils.tensorboard import SummaryWriter
from sklearn.linear_model import Ridge         # ★ 岭回归融合

# ---------------- 1. Backbone ------------------------------------------------
class CifarResNet18(nn.Module):
    """ResNet-18 (conv3-s1, 无 max-pool)，输出 512-d 特征"""
    def __init__(self):
        super().__init__()
        from torchvision.models.resnet import BasicBlock
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)
        self.layer1 = self._block(BasicBlock, 64, 2)
        self.layer2 = self._block(BasicBlock, 128, 2, 2)
        self.layer3 = self._block(BasicBlock, 256, 2, 2)
        self.layer4 = self._block(BasicBlock, 512, 2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = 512
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def _block(self, blk, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * blk.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * blk.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * blk.expansion),
            )
        layers = [blk(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * blk.expansion
        for _ in range(1, blocks):
            layers.append(blk(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.avgpool(x).flatten(1)       # [B,512]

# ---------------- 2. Experts & utilities ------------------------------------
class ExpertHead(nn.Module):
    """两层 MLP 512→512→C"""
    def __init__(self, d, ncls):
        super().__init__()
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, ncls)
    def forward(self, z):
        return self.fc2(F.relu(self.fc1(z), inplace=True))

def greedy_diverse_select(probs: torch.Tensor, n: int) -> torch.Tensor:
    """
    多样性贪婪选专家 (差异①)
    probs: [K,C] softmax  → 返回 bool mask[K]
    """
    K = probs.size(0)
    conf = probs.max(-1).values   # [K]
    sel = torch.zeros(K, dtype=torch.bool, device=probs.device)
    sel[torch.argmax(conf)] = True
    while sel.sum() < n:
        remain = (~sel).nonzero(as_tuple=False).squeeze(1)
        cand   = probs[remain]            # [r,C]
        sel_vec= probs[sel]               # [s,C]
      
        dist = 1 - F.cosine_similarity(cand.unsqueeze(1), sel_vec.unsqueeze(0), dim=-1).max(1).values
        sel[remain[torch.argmax(dist)]] = True
    return sel

def load_balance_loss(gates: torch.Tensor):
    """MoE 负载均衡正则 (差异②)"""
    return ((gates.mean(0) - 1. / gates.size(1)) ** 2).sum()

# ---------------- 3. Branch & DUET-L ----------------------------------------
class Branch(nn.Module):
    def __init__(self, d, ncls, K):
        super().__init__()
        self.experts = nn.ModuleList(ExpertHead(d, ncls) for _ in range(K))

    def forward(self, z: torch.Tensor, n_exp: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        z      : [B,d]; n_exp : [B] 每样本专家数
        返回 logits[B,C], gates[B,K]
        """
        B = z.size(0)
        logits_all = torch.stack([e(z) for e in self.experts], 1)  # [B,K,C]
        probs = F.softmax(logits_all, -1)
        conf  = probs.max(-1).values                               # [B,K]
        gates = torch.zeros_like(conf)

        for b in range(B):
            mask = greedy_diverse_select(probs[b], n_exp[b].item())
            sel_conf = conf[b] * mask
            w = F.softmax(sel_conf.masked_fill(~mask, -1e4) / 0.2, 0)
            gates[b] = w
        logits = (logits_all * gates.unsqueeze(-1)).sum(1)
        return logits, gates

class DuetL(nn.Module):
    def __init__(self, ncls=10, K=6, N_bar=3):
        super().__init__()
        self.backbone = CifarResNet18()
        self.probe   = nn.Linear(self.backbone.out_dim, ncls)  # 轻探针
        self.N_bar   = N_bar
        self.u = Branch(self.backbone.out_dim, ncls, K)
        self.r = Branch(self.backbone.out_dim, ncls, K)

    @staticmethod
    def _entropy(p: torch.Tensor) -> torch.Tensor:
            return -(p * p.clamp_min(1e-9).log()).sum(-1)

    def _n_exp(self, H: torch.Tensor) -> torch.Tensor:
        tau = H.median()
        # H 和返回值 shape 一致，元素 ∈ {N_bar, N_bar+1}
        return torch.where(H <= tau, self.N_bar, self.N_bar + 1)

    # ---------------- 推理接口：自动使用 Ridge 融合 ------------------
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        推理阶段：先用 probe 估计熵，再决定每个样本的 N_exp(x)。
        """
        z = self.backbone(x)   # [B, d]
        with torch.no_grad():
            logits_probe = self.probe(z)                      # [B, C]
            p_probe = F.softmax(logits_probe, dim=-1)         # [B, C]
            H = self._entropy(p_probe)                        # [B]
            n_exp = self._n_exp(H)                            # [B]

        log_u, _ = self.u(z, n_exp)
        log_r, _ = self.r(z, n_exp)

        if hasattr(self, "W_fus"):
            logits = torch.cat([log_u, log_r], 1) @ self.W_fus.t()
        else:
            logits = 0.5 * (log_u + log_r)
        return logits

    # ---------------- 前向：训练 & 返回 loss ----------------------
    def forward(self, x_u, x_r, cfg: Dict = None):
        """
        x_u, x_r : 两个 view（U-branch / R-branch）
        cfg      : 训练配置字典，包含 target, lambda_bt, use_lb, lambda_M, lambda_probe 等
        """
        # 1) backbone 前向：对 backbone 正常回传梯度
        z_u = self.backbone(x_u)   # [B, d]
        z_r = self.backbone(x_r)   # [B, d]

        # 2) probe：只用 detached 特征，不回传到 backbone
        z_cat_det = torch.cat([z_u.detach(), z_r.detach()], dim=0)   # [2B, d]
        logits_probe = self.probe(z_cat_det)                         # [2B, C]
        p_probe = F.softmax(logits_probe, dim=-1)
        H = self._entropy(p_probe)                                   # [2B]
        n_u, n_r = self._n_exp(H).split(z_u.size(0))                 # [B], [B]

        # 3) 双分支多专家：N_exp(x) 由难度控制
        log_u, g_u = self.u(z_u, n_u)   # [B,C], [B,K]
        log_r, g_r = self.r(z_r, n_r)   # [B,C], [B,K]

        if cfg is None:
            
            return log_u, log_r

        y = cfg['target']                         # [B]
        lambda_bt = cfg['lambda_bt']
        use_lb    = cfg['use_lb']                 #
        lambda_M  = cfg['lambda_M']
        lambda_probe = cfg.get('lambda_probe', 0.0)

        # 4) 分类主损失（两个分支）
        loss = F.cross_entropy(log_u, y) + F.cross_entropy(log_r, y)

        # 5) probe 的监督损失
        if lambda_probe > 0:
            y_rep = torch.cat([y, y], dim=0)          # [2B]
            loss_probe = F.cross_entropy(logits_probe, y_rep)
            loss = loss + lambda_probe * loss_probe

        # 6) Barlow Twins 去冗余：把跨分支相关性整体推向 0
        eps = 1e-6
        zu = (z_u - z_u.mean(0)) / (z_u.std(0) + eps)   # [B,d]
        zr = (z_r - z_r.mean(0)) / (z_r.std(0) + eps)   # [B,d]
        c  = (zu.T @ zr) / z_u.size(0)                  # [d,d]

        # 理论版：对角 + 非对角都 → 0
        on_diag  = (c.diag() ** 2).sum()
        off_diag = (c - torch.diag(c.diag())).pow(2).sum()
        loss = loss + lambda_bt * (on_diag + 5e-3 * off_diag)

        # 7) MoE 负载均衡正则（两个分支）
        if use_lb:
            loss = loss + lambda_M * (load_balance_loss(g_u) + load_balance_loss(g_r))

        return loss

# ---------------- 4. DifficultySampler --------------------------------------
class DifficultySampler(Sampler):
    """λ_RS 融合 (尾类×难度×熵) 概率，与 uniform 混合"""
    def __init__(self, labels: np.ndarray, lambda_rs=0.5):
        self.labels = labels
        self.N = len(labels)
        cls_cnt = np.bincount(labels)
        inv_freq = 1. / np.maximum(cls_cnt[labels], 1)
        self.base = inv_freq / inv_freq.mean()
        self.lambda_rs = lambda_rs
        self.uniform = np.ones(self.N) / self.N
        self.p = self.uniform.copy()

    def update(self, ce_hist: np.ndarray, entropy: np.ndarray):
        """
        ce_hist : 每个样本的 EMA 交叉熵 L_CE(x)
        entropy : 每个样本的熵 H(x)
        """
        q = self.base * ce_hist * (1.0 + entropy)   # w_c^{-1} · CE · (1+H)
        s = q.sum()
        if s <= 0 or np.isnan(s):
            self.p = self.uniform.copy()
            return
        q /= s
        self.p = (1 - self.lambda_rs) * self.uniform + self.lambda_rs * q

    def __iter__(self): return iter(np.random.choice(self.N, self.N, p=self.p))
    def __len__(self):  return self.N

# ---------------- 5. Metrics -------------------------------------------------
class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.sum = self.cnt = 0.
    def update(self, v, n): self.sum += v * n; self.cnt += n
    @property
    def avg(self): return self.sum / max(1, self.cnt)

@torch.no_grad()
def topk(logit, tgt, k=1):
    _, pred = logit.topk(k, 1, True, True)
    return pred.eq(tgt.view(-1, 1)).float().sum().mul_(100. / tgt.size(0))

def seg_split(labels):
    cnt = np.bincount(labels); idx = np.argsort(cnt)[::-1]
    cum = np.cumsum(cnt[idx]) / cnt.sum()
    head = idx[cum <= 0.5]; mid = idx[(cum > 0.5) & (cum <= 0.9)]; tail = idx[cum > 0.9]
    return dict(head=head, mid=mid, tail=tail)

# ---------------- 6. Data ----------------------------------------------------
def cifar_loaders(cfg):
    T_train = transforms.Compose([
        transforms.RandomCrop(32, 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))])
    T_test  = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))])

    Data = datasets.CIFAR10 if cfg['dataset'] == 'cifar10' else datasets.CIFAR100
    # ----- 读取长尾索引 -----
    if cfg.get('lt_dir'):                     
        lt_root = cfg['lt_dir']
    else:                                          
        lt_root = os.path.join(cfg['datapath'],
                               f"{'cifar-10' if cfg['dataset']=='cifar10' else 'cifar-100'}-LT-10")

    idx_file = os.path.join(lt_root, 'indices_train_lt.txt')
    if not os.path.isfile(idx_file):
        raise FileNotFoundError(f'找不到长尾索引 {idx_file}')

    with open(idx_file) as f:
        idx = [int(i) for i in f.read().split()]

    full_train = Data(cfg['datapath'], True, download=True, transform=T_train)
    train_set  = torch.utils.data.Subset(full_train, idx)
    val_set   = Data(cfg['datapath'], False, download=True, transform=T_test)
    if isinstance(train_set, torch.utils.data.Subset):

        full_labels = np.array(train_set.dataset.targets)
        labels = full_labels[train_set.indices]
    else:
        labels = np.array(train_set.targets)

    u_loader = DataLoader(train_set, cfg['batch_size'],
                          sampler=RandomSampler(train_set),
                          num_workers=6, pin_memory=True, drop_last=True)
    diff_sampler = DifficultySampler(labels, cfg['lambda_rs'])
    r_loader = DataLoader(train_set, cfg['batch_size'],
                          sampler=diff_sampler,
                          num_workers=6, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, cfg['batch_size'],
                            shuffle=False, num_workers=6, pin_memory=True)
    return u_loader, r_loader, val_loader, labels, diff_sampler

# ---------------- 7. Ridge Fusion 拟合 --------------------------------------
@torch.no_grad()
def fit_ridge_fusion(model: DuetL, loader: DataLoader, beta: float = 1.0):
    model.eval(); X, Y = [], []
    for x, y in loader:
        x, y = x.cuda(), y.cuda()
        log_u, log_r = model(x, x)    
        X.append(torch.cat([log_u, log_r], 1).cpu())
        Y.append(F.one_hot(y, num_classes=log_u.size(1)).float().cpu())
    X = torch.cat(X).numpy(); Y = torch.cat(Y).numpy()
    coef = Ridge(alpha=beta, fit_intercept=False).fit(X, Y).coef_   # [C,2C]
    model.W_fus = torch.tensor(coef, dtype=torch.float32, device="cuda")
    print(">> Ridge-fusion weights fitted.")

# ---------------- 8. Train & Eval -------------------------------------------
@torch.no_grad()
def evaluate(model: DuetL, loader, seg_map, tb, ep, amp=False):
    model.eval()
    top1 = AverageMeter()
    per_cls = defaultdict(list)
    all_logits, all_labels = [], []

    for x, y in loader:
        x, y = x.cuda(), y.cuda()
        with autocast('cuda', enabled=amp):
            logits = model.predict(x)
        top1.update(topk(logits, y).item(), x.size(0))

        all_logits.append(logits.cpu())
        all_labels.append(y.cpu())

        preds = logits.argmax(1).cpu()
        for t, p in zip(y.cpu(), preds):
            per_cls[t.item()].append(int(t == p))
        # ---- 分段精度 ----
    # ---- helper: segment指标 ----
    def seg_metrics(ids):
        mask = torch.isin(torch.cat(all_labels), torch.tensor(ids))
        if mask.sum() == 0:
            return dict(acc=float('nan'), auc=None, gmean=float('nan'), f1=float('nan'))
        y_seg = torch.cat(all_labels)[mask]
        p_seg = torch.cat(all_logits)[mask]
        acc = (p_seg.argmax(1) == y_seg).float().mean().item() * 100
        try:
            auc = roc_auc_score(
                y_seg.numpy(), F.softmax(p_seg, -1).numpy(),
                multi_class='ovr', average='macro')
        except ValueError:
            auc = float('nan')    
        f1 = f1_score(y_seg.numpy(), p_seg.argmax(1).numpy(), average='macro')
        # ---------- gmean ----------
        cm = confusion_matrix(y_seg, p_seg.argmax(1), labels=ids)
        rec = np.diag(cm) / cm.sum(1).clip(min=1)
        gmean = float(np.exp(np.log(np.clip(rec,1e-12,1)).mean()))
        return dict(acc=acc, auc=auc, gmean=gmean, f1=f1)

    head_m = seg_metrics(seg_map['head'])
    mid_m  = seg_metrics(seg_map['mid'])
    tail_m = seg_metrics(seg_map['tail'])

    # ---- overall ----
    Y   = torch.cat(all_labels)
    P   = torch.cat(all_logits)
    try:
        auc_all = roc_auc_score(
            Y.numpy(), F.softmax(P, -1).numpy(), multi_class='ovr', average='macro')
    except ValueError:
        auc_all = float('nan')
    f1_all  = f1_score(Y.numpy(), P.argmax(1).numpy(), average='macro')
    # ---- overall gmean ----
    cm_all  = confusion_matrix(Y.numpy(), P.argmax(1).numpy(), labels=np.arange(P.size(1)))
    rec_all = np.diag(cm_all) / cm_all.sum(1).clip(min=1)
    gmean_all = float(np.exp(np.log(np.clip(rec_all,1e-12,1)).mean()))
    metr = dict(
        acc_all = top1.avg,
        auc_all = auc_all,
        gmean_all = gmean_all,
        f1_all  = f1_all,
        head_acc=head_m['acc'], head_auc=head_m['auc'], head_gmean=head_m['gmean'], head_f1=head_m['f1'],
        mid_acc =mid_m['acc'],  mid_auc =mid_m['auc'],  mid_gmean=mid_m['gmean'], mid_f1=mid_m['f1'],
        tail_acc=tail_m['acc'], tail_auc=tail_m['auc'], tail_gmean=tail_m['gmean'], tail_f1=tail_m['f1'],

    )

    # ---- AUC & F1 ----
    Y   = torch.cat(all_labels)
    P   = torch.cat(all_logits)
    metr['auc'] = roc_auc_score(
        Y.numpy(), F.softmax(P, -1).numpy(), multi_class='ovr', average='macro')
    metr['f1']  = f1_score(
        Y.numpy(), P.argmax(1).numpy(), average='macro')

    # ---- TensorBoard 持续记录 ----
    for k, v in metr.items():
        # 只记录可用、有限的数值；跳过 None / nan / inf
        if v is not None and np.isfinite(v):
            tb.add_scalar('val/' + k, v, ep)

    return metr

def train(cfg):
    torch.cuda.set_device(cfg.get('gpu', 0)); torch.backends.cudnn.benchmark = True
    # ========= ① 结果输出根目录 =========

    output_dir = pathlib.Path(cfg.get("lt_dir", "."))
    output_dir.mkdir(parents=True, exist_ok=True)   # 若不存在则创建

    u_loader, r_loader, val_loader, labels, diff_sp = cifar_loaders(cfg)
    seg_map = seg_split(labels)
    model = DuetL(cfg['num_classes'], cfg['K'], cfg['N_bar']).cuda()
    opt = torch.optim.SGD(model.parameters(), cfg['lr'], 0.9, weight_decay=5e-4)
    tb  = SummaryWriter(comment=cfg['dataset'])
    best_tail = 0.
    try:
        scaler = torch.amp.GradScaler(enabled=cfg['amp'])
    except AttributeError:      # <2.1 回退
        scaler = torch.cuda.amp.GradScaler(enabled=cfg['amp'])
    for ep in range(cfg['epochs']):
        tic = time.time()  
        model.train(); meter=AverageMeter(); r_iter=iter(r_loader)
        pbar = tqdm.tqdm(u_loader, desc=f'E{ep}')
        for x_u, y in pbar:
            try: x_r,_=next(r_iter)
            except StopIteration: r_iter=iter(r_loader); x_r,_=next(r_iter)
            x_u,x_r,y = x_u.cuda(),x_r.cuda(),y.cuda()
                        # ---------- AMP 前向 ----------
            with autocast('cuda', enabled=cfg['amp']):
                loss = model(
                    x_u, x_r,
                    dict(
                        target=y,
                        lambda_bt=cfg['lambda_bt'],
                        use_lb=cfg['use_load_balance'],
                        lambda_M=cfg['lambda_M'],
                        lambda_probe=cfg.get('lambda_probe', 0.0),
                    )
                )

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

            meter.update(loss.item(), x_u.size(0))
            pbar.set_postfix(loss=f'{meter.avg:.3f}')
        tb.add_scalar('train/loss', meter.avg, ep)
        metr = evaluate(model, val_loader, seg_map, tb, ep, amp=cfg['amp'])
        write_csv(cfg | {"epochs": ep + 1}, metr, output_dir / "results.csv")
        write_result(cfg | {"epochs": ep + 1}, metr, output_dir / "results.md")

        # --------- 根据 CE + 熵 更新 DifficultySampler ---------
        if cfg['lambda_rs'] > 0:
            ce_epoch   = np.zeros(len(labels), dtype=np.float32)
            entr_epoch = np.zeros(len(labels), dtype=np.float32)

            model.eval()
            with torch.no_grad():
                for idx, (img, lbl) in enumerate(u_loader.dataset):
                    img = img.unsqueeze(0).cuda()
                    lbl = torch.as_tensor([lbl], device=img.device)

                    z = model.backbone(img)                  # [1,d]
                    logits_probe = model.probe(z)           # [1,C]
                    p = F.softmax(logits_probe, -1)[0]      # [C]

                    # 熵 H(x)
                    entr_epoch[idx] = (-(p * p.clamp_min(1e-9).log()).sum()).item()
                    # 交叉熵 L_CE(x) = -log p_y
                    ce_epoch[idx] = F.cross_entropy(logits_probe, lbl).item()

            # EMA 累积 per-sample CE
            if not hasattr(diff_sp, 'ce_hist'):
                diff_sp.ce_hist = ce_epoch.copy()
            diff_sp.ce_hist = 0.9 * diff_sp.ce_hist + 0.1 * ce_epoch

            # 用 CE_hist + 熵 更新采样分布
            diff_sp.update(diff_sp.ce_hist, entr_epoch)
            model.train()
        if metr['tail_acc'] > best_tail:
            best_tail = metr['tail_acc']
            torch.save(model.state_dict(), output_dir / "best_tail.pth")
        toc = time.time() - tic
        print(
            f"Epoch {ep+1}/{cfg['epochs']} | "
            f"train_loss {meter.avg:.4f} | "
            f"val {fmt(metr['acc_all'])}/{fmt(metr['auc_all'])}/{fmt(metr['gmean_all'])}/{fmt(metr['f1_all'])} | "
            f"H {fmt(metr['head_acc'])}/{fmt(metr['head_auc'])}/{fmt(metr['head_gmean'])}/{fmt(metr['head_f1'])} | "
            f"M {fmt(metr['mid_acc'])}/{fmt(metr['mid_auc'])}/{fmt(metr['mid_gmean'])}/{fmt(metr['mid_f1'])} | "
            f"T {fmt(metr['tail_acc'])}/{fmt(metr['tail_auc'])}/{fmt(metr['tail_gmean'])}/{fmt(metr['tail_f1'])} | "
            f"t {toc:.1f}s")
    
    # -------- 训练结束：拟合 Ridge 融合权重 --------
    fit_ridge_fusion(model, val_loader)
    torch.save(model.state_dict(), output_dir / "final.pth")
    write_result(cfg, metr, file_path=output_dir / "results.md")
    write_csv(cfg, metr, file_path=output_dir / "results.csv")
    tb.close()

# ============================================================
# util: 写入 Markdown 结果表
# ============================================================
def fmt(x: float) -> str:
    return "nan" if (x is None or not np.isfinite(x)) else f"{x:.4f}"
# ---------------- CSV 版本 ----------------
def write_csv(cfg, metr, file_path='results.csv'):
    path = pathlib.Path(file_path)
    if not path.exists():
        path.write_text(
            "method,seed,epoch,acc,auc,gmean,f1,"
            "head_acc,head_auc,head_gmean,head_f1,"
            "mid_acc,mid_auc,mid_gmean,mid_f1,"
            "tail_acc,tail_auc,tail_gmean,tail_f1,file\n")


    with open(path, 'a', newline='', encoding='utf-8') as f:
        f.write(
            f"{cfg.get('method','duetl')},{cfg['seed']},{cfg['epochs']},"
            f"{fmt(metr['acc_all'])},{fmt(metr['auc_all'])},{fmt(metr['gmean_all'])},{fmt(metr['f1_all'])},"
            f"{fmt(metr['head_acc'])},{fmt(metr['head_auc'])},{fmt(metr['head_gmean'])},{fmt(metr['head_f1'])},"
            f"{fmt(metr['mid_acc'])},{fmt(metr['mid_auc'])},{fmt(metr['mid_gmean'])},{fmt(metr['mid_f1'])},"
            f"{fmt(metr['tail_acc'])},{fmt(metr['tail_auc'])},{fmt(metr['tail_gmean'])},{fmt(metr['tail_f1'])},final.pth\n")

def write_result(cfg, metr, file_path='results.md'):
    path = pathlib.Path(file_path)
    if not path.exists():
        path.write_text(
            "| method | seed | epoch | acc | auc | gmean | f1 | "
            "head_acc | head_auc | head_gmean | head_f1 | "
            "mid_acc | mid_auc | mid_gmean | mid_f1 | "
            "tail_acc | tail_auc | tail_gmean | tail_f1 | file |\n"
            "|--------|------|-------|-----|-----|----|"
            "---------|---------|--------|"
            "--------|---------|--------|"
            "---------|---------|--------|------|\n")
    with open(path, 'a', encoding='utf-8') as f:
        f.write(
            f"| {cfg.get('method','duetl')} | {cfg['seed']} | {cfg['epochs']} | "
            f"{fmt(metr['acc_all'])} | {fmt(metr['auc_all'])} | {fmt(metr['gmean_all'])} | {fmt(metr['f1_all'])} | "
            f"{fmt(metr['head_acc'])} | {fmt(metr['head_auc'])} | {fmt(metr['head_gmean'])} | {fmt(metr['head_f1'])} | "
            f"{fmt(metr['mid_acc'])} | {fmt(metr['mid_auc'])} | {fmt(metr['mid_gmean'])} | {fmt(metr['mid_f1'])} | "
            f"{fmt(metr['tail_acc'])} | {fmt(metr['tail_auc'])} | {fmt(metr['tail_gmean'])} | {fmt(metr['tail_f1'])} | final.pth |\n")

    

# ---------------- 9. Main ----------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', type=str, default=None,
                        help='可选：YAML 配置路径；若提供则忽略其它 CLI 超参')
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10','cifar100'])
    parser.add_argument('--datapath', required=False, default='./data')
    parser.add_argument('--lt_dir', type=str, default=None,
                    help='可选：指定长尾子目录，如 L:\\appp\\...\\cifar-10-LT-50')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--K', type=int, default=6)
    parser.add_argument('--N_bar', type=int, default=3)
    parser.add_argument('--lambda_rs', type=float, default=0.6)
    parser.add_argument('--lambda_bt', type=float, default=0.02)
    parser.add_argument('--use_load_balance', type=str, default='false',
                        help='true/false')
    parser.add_argument('--lambda_M', type=float, default=0.01)
    parser.add_argument('--lambda_probe', type=float, default=0.0,
                        help='probe 的监督损失权重')
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--amp', type=str, default='true',
                        help='开启自动混合精度 (AMP)')
    args = parser.parse_args()


    if args.cfg is not None:
        if yaml is None:
            raise RuntimeError("未安装 PyYAML，无法解析 --cfg。可改用纯 CLI 方式或 pip install pyyaml。")
        cfg = yaml.safe_load(open(args.cfg, 'r', encoding='utf-8'))
    else:
        cfg = {
            'dataset': args.dataset,
            'datapath': args.datapath,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'num_classes': args.num_classes,
            'K': args.K,
            'N_bar': args.N_bar,
            'lambda_rs': args.lambda_rs,
            'lambda_bt': args.lambda_bt,
            'use_load_balance': str(args.use_load_balance).lower() in ['true','1','yes','y'],
            'lambda_M': args.lambda_M,
            'lambda_probe': args.lambda_probe,
            'seed': args.seed,
            'gpu': args.gpu,
            'amp': str(args.amp).lower() in ['true', '1', 'yes', 'y'],
        }
    train(cfg)



