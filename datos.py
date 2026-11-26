# Datos: CIFAR-10/100 completo en la GPU y aumento de datos por lote
import os
import torch
import torch.nn.functional as F
import config as cfg

MEDIAS = {"cifar10": (0.4914, 0.4822, 0.4465), "cifar100": (0.5071, 0.4865, 0.4409)}
STDS   = {"cifar10": (0.2470, 0.2435, 0.2616), "cifar100": (0.2673, 0.2564, 0.2762)}


def cargar_cifar(nombre, sub_train=None, sub_test=None, val_frac=0.1, semilla_split=1234):
    # devuelve los datos como tensores uint8 en el dispositivo (train con padding de 4 pixeles)
    nc = 10 if nombre == "cifar10" else 100
    if os.environ.get("PROXSAM_DATOS_FALSOS") == "1":
        # solo para probar el codigo sin internet: datos aleatorios
        print("OJO: se usan datos aleatorios (PROXSAM_DATOS_FALSOS=1), los resultados no significan nada")
        g = torch.Generator().manual_seed(0)
        Xtr = torch.randint(0, 256, (50000, 3, 32, 32), dtype=torch.uint8, generator=g)
        Ytr = torch.randint(0, nc, (50000,), generator=g)
        Xte = torch.randint(0, 256, (10000, 3, 32, 32), dtype=torch.uint8, generator=g)
        Yte = torch.randint(0, nc, (10000,), generator=g)
    else:
        import torchvision
        cls = torchvision.datasets.CIFAR10 if nombre == "cifar10" else torchvision.datasets.CIFAR100
        os.makedirs(cfg.DATOS, exist_ok=True)
        tr = cls(cfg.DATOS, train=True, download=True)
        te = cls(cfg.DATOS, train=False, download=True)
        Xtr = torch.from_numpy(tr.data).permute(0, 3, 1, 2).contiguous()
        Ytr = torch.tensor(tr.targets)
        Xte = torch.from_numpy(te.data).permute(0, 3, 1, 2).contiguous()
        Yte = torch.tensor(te.targets)
    g = torch.Generator().manual_seed(semilla_split)
    perm = torch.randperm(len(Xtr), generator=g)
    nval = int(val_frac * len(Xtr))
    iva, itr = perm[:nval], perm[nval:]
    if sub_train:
        itr = itr[:sub_train]
        iva = iva[: max(500, sub_train // 8)]
    if sub_test:
        Xte, Yte = Xte[:sub_test], Yte[:sub_test]
    dev = cfg.DEVICE
    Xtr_pad = F.pad(Xtr[itr], (4, 4, 4, 4))
    return dict(Xtr=Xtr_pad.to(dev), Ytr=Ytr[itr].to(dev),
                Xva=Xtr[iva].to(dev), Yva=Ytr[iva].to(dev),
                Xte=Xte.to(dev), Yte=Yte.to(dev),
                media=torch.tensor(MEDIAS[nombre], device=dev).view(1, 3, 1, 1),
                std=torch.tensor(STDS[nombre], device=dev).view(1, 3, 1, 1),
                n_clases=nc, nombre=nombre)


_AR32 = {}


def aumentar(Xpad_b, d, cutout=16, p_cutout=0.5):
    # recorte aleatorio, flip y cutout de un lote, todo en el dispositivo
    B, dev = Xpad_b.shape[0], Xpad_b.device
    ar = _AR32.setdefault(dev, torch.arange(32, device=dev))
    i = torch.randint(0, 9, (B,), device=dev)
    j = torch.randint(0, 9, (B,), device=dev)
    filas, cols = i[:, None] + ar[None, :], j[:, None] + ar[None, :]
    x = Xpad_b[torch.arange(B, device=dev)[:, None, None], :, filas[:, :, None], cols[:, None, :]]
    x = x.permute(0, 3, 1, 2)
    volt = torch.rand(B, device=dev) < 0.5
    x = torch.where(volt[:, None, None, None], x.flip(3), x)
    x = (x.float() / 255.0 - d["media"]) / d["std"]
    if cutout:
        # cuadrado de lado `cutout` (recortado en los bordes), con probabilidad p_cutout
        h = cutout // 2
        arr = torch.randint(-h, 32 - h, (B, 1, 1), device=dev)
        izq = torch.randint(-h, 32 - h, (B, 1, 1), device=dev)
        fil, col = ar.view(1, 32, 1), ar.view(1, 1, 32)
        dentro = (fil >= arr) & (fil < arr + cutout) & (col >= izq) & (col < izq + cutout)
        dentro = dentro & (torch.rand(B, 1, 1, device=dev) < p_cutout)
        x = x * (~dentro).unsqueeze(1)
    return x.contiguous(memory_format=torch.channels_last)


def normalizar(X, d):
    return ((X.float() / 255.0 - d["media"]) / d["std"]).contiguous(memory_format=torch.channels_last)


_cache = {}


def datos_para(nombre):
    P = cfg.P()
    clave = (nombre, P["sub_train"], P["sub_test"])
    if clave not in _cache:
        _cache.clear()
        if cfg.DEVICE == "cuda":
            torch.cuda.empty_cache()
        _cache[clave] = cargar_cifar(nombre, P["sub_train"], P["sub_test"], cfg.BASE["val_frac"])
        d = _cache[clave]
        print(f"  datos {nombre} en {cfg.DEVICE}: train {len(d['Xtr'])}, val {len(d['Xva'])}, test {len(d['Xte'])}", flush=True)
    return _cache[clave]
