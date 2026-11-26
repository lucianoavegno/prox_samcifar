# Optimizador: una sola clase para todos los metodos, lo que cambia es el punto perturbado
#   sgd       sin perturbacion (con lambda > 0 es Prox-SGD)
#   sam       w + rho * g1/||g1||
#   sam_map   w + rho * G_t(w)/||G_t(w)||
#   sam_rec   como sam, con el radio recortado min(rho, nu*||G_tbar(w)||/L)
#   usam      w + rho * g1
#   usam_corr w + rho * G_tbar(w)
import torch
import torch.nn as nn
import torch.nn.functional as F
import config as cfg


def umbral_suave_(p, tau):
    # soft-thresholding in-place
    p.copy_(torch.sign(p) * (p.abs() - tau).clamp_(min=0))


def norma_global(tensores):
    return torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(t) for t in tensores]))


class ProxSAM:
    def __init__(self, modelo, metodo="sam", momentum=0.0, wd=1e-3, nu=0.5, L0=100.0, escala0=2.0 ** 14):
        self.m, self.metodo, self.mu, self.wd = modelo, metodo, momentum, wd
        self.params = [p for p in modelo.parameters() if p.requires_grad]
        self.podable = [p.dim() > 1 for p in self.params]
        self.pod = [p for p, b in zip(self.params, self.podable) if b]
        self.buf = [torch.zeros_like(p) for p in self.params]
        self.S, self.buenos, self.descartados, self.seguidos = escala0, 0, 0, 0
        self.nu, self.L_hat = nu, L0
        self.bns = [mod for mod in modelo.modules() if isinstance(mod, nn.modules.batchnorm._BatchNorm)]
        self.usa_amp = cfg.DEVICE == "cuda"
        self.reiniciar_stats()

    def reiniciar_stats(self):
        # sumas por epoca: radio efectivo y cociente ||g(y) - g(w)|| / ||y - w||
        self.s_rho, self.s_sec, self.n_stats = 0.0, 0.0, 0

    def _grad(self, x, y):
        for p in self.params:
            p.grad = None
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.usa_amp):
            out = self.m(x)
            perdida = F.cross_entropy(out, y)
        # la perdida se multiplica por S y los gradientes se dividen por S (precision mixta)
        (perdida * self.S).backward()
        g = [p.grad for p in self.params]
        with torch.no_grad():
            for gi in g:
                gi.div_(self.S)
            ok = torch.isfinite(norma_global(g))
            if self.wd:
                for gi, p in zip(g, self.params):
                    gi.add_(p, alpha=self.wd)
        return perdida.detach(), out.detach(), ok

    @torch.no_grad()
    def _gmap(self, g, t, lam):
        # mapeo de gradiente proximal; en los parametros sin norma 1 es el gradiente
        G = []
        for p, gi, b in zip(self.params, g, self.podable):
            if b and lam > 0:
                z = p - t * gi
                G.append((p - torch.sign(z) * (z.abs() - t * lam).clamp(min=0)) / t)
            else:
                G.append(gi.clone())
        return G

    def _bn_congelar(self, si):
        for bn in self.bns:
            if si:
                bn._mom_backup, bn.momentum = bn.momentum, 0.0
            else:
                bn.momentum = bn._mom_backup

    def _descartar(self):
        # paso con inf o nan: no se actualiza nada y se baja la escala
        self.descartados += 1
        self.seguidos += 1
        if self.usa_amp and self.seguidos >= 20:
            # 20 pasos seguidos: no es la escala, se sigue en precision simple
            self.usa_amp, self.S = False, 1.0
            print("    (precision mixta desactivada en esta corrida)", flush=True)
        elif self.usa_amp:
            self.S = max(self.S / 2, 1.0)
        elif self.seguidos >= 100:
            raise RuntimeError("gradientes no finitos tambien en precision simple: el entrenamiento divergio")

    def paso(self, x, y, lr, rho, lam, tbar):
        # primera pasada: gradiente en w
        perdida, out, ok = self._grad(x, y)
        if not bool(ok):
            # gradiente no finito en w: se descarta antes de perturbar (si no, los pesos quedan en nan)
            self._descartar()
            return perdida, out, False
        e, g1, norma_e = None, None, None
        if self.metodo != "sgd" and rho > 0:
            with torch.no_grad():
                g1 = [p.grad for p in self.params]
                if self.metodo in ("sam", "sam_rec", "usam"):
                    d = g1
                elif self.metodo == "sam_map":
                    d = self._gmap(g1, lr, lam)
                elif self.metodo == "usam_corr":
                    d = self._gmap(g1, tbar, lam)
                if self.metodo in ("usam", "usam_corr"):
                    e = [di * rho for di in d]
                    norma_e = rho * norma_global(d)
                else:
                    r = rho
                    if self.metodo == "sam_rec":
                        nG = norma_global(self._gmap(g1, tbar, lam))
                        r = torch.clamp(self.nu * nG / self.L_hat, max=rho)
                    coef = r / (norma_global(d) + 1e-12)
                    e = [di * coef for di in d]
                    norma_e = r
                for p, ei in zip(self.params, e):
                    p.add_(ei)
            # segunda pasada: gradiente en el punto perturbado (sin actualizar BatchNorm)
            self._bn_congelar(True)
            perdida2, _, ok2 = self._grad(x, y)
            self._bn_congelar(False)
            ok = ok & ok2
        with torch.no_grad():
            # se vuelve a w
            if e is not None:
                for p, ei in zip(self.params, e):
                    p.sub_(ei)
            # si hubo inf o nan se descarta el paso y se baja la escala
            if not bool(ok):
                self._descartar()
                return perdida, out, False
            self.seguidos = 0
            g = [p.grad for p in self.params]
            if g1 is not None:
                # estimacion de L con los dos gradientes del paso
                norma_e = float(norma_e)
                sec = float(norma_global([a - b for a, b in zip(g, g1)])) / (norma_e + 1e-12)
                self.L_hat = max(0.999 * self.L_hat, sec)
                self.s_rho += norma_e
                self.s_sec += sec
                self.n_stats += 1
            # paso (con momentum si corresponde) y despues el prox
            if self.mu:
                for b, gi, p in zip(self.buf, g, self.params):
                    b.mul_(self.mu).add_(gi)
                    p.add_(b, alpha=-lr)
            else:
                for gi, p in zip(g, self.params):
                    p.add_(gi, alpha=-lr)
            if lam > 0:
                # con momentum el umbral lleva el factor 1/(1 - beta)
                tau = lr * lam / (1.0 - self.mu)
                for p in self.pod:
                    umbral_suave_(p, tau)
        self.buenos += 1
        if self.usa_amp and self.buenos % 2000 == 0:
            self.S = min(self.S * 2, 2.0 ** 16)
        return perdida, out, True

    def estado(self):
        return dict(buf=[b.detach().cpu() for b in self.buf], S=self.S, buenos=self.buenos,
                    descartados=self.descartados, L_hat=self.L_hat, amp=self.usa_amp,
                    s_rho=self.s_rho, s_sec=self.s_sec, n_stats=self.n_stats)

    def cargar(self, st):
        for b, s in zip(self.buf, st["buf"]):
            b.copy_(s.to(b.device))
        self.S, self.buenos, self.descartados = st["S"], st["buenos"], st["descartados"]
        self.L_hat = st.get("L_hat", self.L_hat)
        self.usa_amp = st.get("amp", self.usa_amp) and cfg.DEVICE == "cuda"
        if self.usa_amp:
            self.S = min(max(self.S, 1.0), 2.0 ** 16)
        self.s_rho, self.s_sec, self.n_stats = st.get("s_rho", 0.0), st.get("s_sec", 0.0), st.get("n_stats", 0)
