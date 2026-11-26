# Reglas de paso, radios y metricas
import copy, math
import torch
import torch.nn.functional as F
import config as cfg
from datos import normalizar
from optimizador import norma_global


def paso_lr(sched, n, eta1):
    # reglas del paper, n es la epoca (parte en 1)
    if sched == "const":
        return eta1
    if sched == "dim1":
        return eta1 / n
    if sched == "dim2":
        return eta1 / n ** 0.5001
    if sched == "dim3":
        m = n // 5 + 2
        return eta1 / (m * math.log(m))
    raise ValueError(sched)


def radio(sched, n, rho0, delta=0.01):
    if sched == "const":
        return rho0
    if sched == "casi":
        return rho0 / n ** 0.001
    if sched == "hip31":
        return rho0 / n ** (0.25 + delta)
    if sched == "paper9":
        return rho0 / n ** (0.5 + delta)
    if sched == "cero":
        return 0.0
    raise ValueError(sched)


@torch.no_grad()
def evaluar(modelo, X, Y, d, bs=1000):
    modelo.eval()
    ok, tot, perd = 0, 0, 0.0
    for i in range(0, len(X), bs):
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=cfg.DEVICE == "cuda"):
            out = modelo(normalizar(X[i:i + bs], d))
        perd += F.cross_entropy(out.float(), Y[i:i + bs], reduction="sum").item()
        ok += (out.argmax(1) == Y[i:i + bs]).sum().item()
        tot += len(X[i:i + bs])
    modelo.train()
    return 100.0 * ok / tot, perd / tot


@torch.no_grad()
def densidad(modelo):
    nz = tot = 0
    for p in modelo.parameters():
        if p.dim() > 1:
            nz += (p != 0).sum().item()
            tot += p.numel()
    return nz / tot


@torch.no_grad()
def densidad_por_capa(modelo):
    return {n: (p != 0).float().mean().item() for n, p in modelo.named_parameters() if p.dim() > 1}


def certificado(modelo, Xc, Yc, d, lam, tbar, wd, bs=500):
    # ||G_tbar(w)||, ||grad f(w)|| y dist(0, grad f(w) + subdiferencial de h)
    # con el gradiente completo sobre (Xc, Yc), BatchNorm en modo eval
    modelo.eval()
    params = [p for p in modelo.parameters() if p.requires_grad]
    for p in params:
        p.grad = None
    for i in range(0, len(Xc), bs):
        perd = F.cross_entropy(modelo(normalizar(Xc[i:i + bs], d)), Yc[i:i + bs], reduction="sum") / len(Xc)
        perd.backward()
    with torch.no_grad():
        g = [p.grad + wd * p for p in params]
        G, R = [], []
        for p, gi in zip(params, g):
            if p.dim() > 1 and lam > 0:
                z = p - tbar * gi
                G.append((p - torch.sign(z) * (z.abs() - tbar * lam).clamp(min=0)) / tbar)
                R.append(torch.where(p != 0, gi + lam * torch.sign(p), (gi.abs() - lam).clamp(min=0)))
            else:
                G.append(gi)
                R.append(gi)
        cert, ngrad, dist = norma_global(G).item(), norma_global(g).item(), norma_global(R).item()
    for p in params:
        p.grad = None
    modelo.train()
    return cert, ngrad, dist


def autovalor_hessiano(modelo, X, Y, d, iters=20, bs=1000):
    # autovalor dominante del hessiano con el metodo de las potencias
    modelo.eval()
    params = [p for p in modelo.parameters() if p.requires_grad]
    x, y = normalizar(X[:bs], d), Y[:bs]
    perd = F.cross_entropy(modelo(x), y)
    gr = torch.autograd.grad(perd, params, create_graph=True)
    v = [torch.randn_like(p) for p in params]
    nv = norma_global(v)
    v = [vi / nv for vi in v]
    lam_max = 0.0
    for _ in range(iters):
        Hv = torch.autograd.grad(gr, params, grad_outputs=v, retain_graph=True)
        lam_max = sum((h * vi).sum() for h, vi in zip(Hv, v)).item()
        nh = norma_global(Hv)
        v = [h / (nh + 1e-12) for h in Hv]
    modelo.train()
    return lam_max


@torch.no_grad()
def robustez_poda(modelo, d, fracs=(0.0, 0.5, 0.7, 0.8, 0.9, 0.95)):
    # poda por magnitud sin reentrenar, y precision de prueba
    respaldo = copy.deepcopy(modelo.state_dict())
    pod = [p for p in modelo.parameters() if p.dim() > 1]
    todos = torch.cat([p.abs().flatten() for p in pod])
    res = {}
    for f in fracs:
        modelo.load_state_dict(respaldo)
        if f > 0:
            k = int(f * todos.numel())
            umbral = torch.kthvalue(todos.float().cpu(), max(k, 1)).values.item()
            for p in pod:
                p.mul_((p.abs() > umbral).to(p.dtype))
        res[f] = evaluar(modelo, d["Xte"], d["Yte"], d)[0]
    modelo.load_state_dict(respaldo)
    return res
