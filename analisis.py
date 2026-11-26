# Tablas y figuras. Lee lo que haya en la carpeta de resultados, asi que se puede correr con resultados parciales.
# En Spyder: correr celda por celda (Ctrl+Enter) o completo con F5. Las figuras quedan en resultados/figuras/.
# %% lectura de resultados
import os, sys, glob, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd())
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
import config as cfg

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
mpl.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm", "font.size": 10, "axes.titlesize": 10.5,
    "legend.fontsize": 8.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
    "grid.color": "#d9d8d4", "grid.linestyle": ":", "grid.linewidth": 0.6, "lines.linewidth": 1.6,
    "figure.dpi": 110, "savefig.bbox": "tight"})
C1, C2, C3, C4, C5, C6, C7, C8 = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GRIS, TINTA, TINTA2 = "#8a8985", "#0b0b0b", "#52514e"
FIG = os.path.join(cfg.RAIZ, "figuras")
os.makedirs(FIG, exist_ok=True)


def mostrar(df):
    print(df.to_string() if hasattr(df, "to_string") else df)
    print()


def guardar(fig, nombre):
    fig.savefig(os.path.join(FIG, nombre + ".pdf"))
    fig.savefig(os.path.join(FIG, nombre + ".png"), dpi=160)


def cargar_resultados():
    # LOG: una fila por epoca de cada corrida; FIN: una fila por corrida terminada
    logs, fins = [], []
    for carpeta in sorted(glob.glob(os.path.join(cfg.RAIZ, "*"))):
        f_log, f_fin, f_cfg = [os.path.join(carpeta, n) for n in ("log.csv", "final.json", "config.json")]
        if not (os.path.exists(f_log) and os.path.exists(f_cfg)):
            continue
        df = pd.read_csv(f_log)
        if not len(df):
            continue
        c = json.load(open(f_cfg))["config"]
        df["id"] = os.path.basename(carpeta)
        for k, v in c.items():
            df[k] = v
        logs.append(df)
        if os.path.exists(f_fin):
            fin = json.load(open(f_fin))
            fila = {**fin["config"], **{k: v for k, v in fin.items() if k != "config"}}
            f_ext = os.path.join(carpeta, "extras.json")
            if os.path.exists(f_ext):
                fila.update(json.load(open(f_ext)))
            fins.append(fila)
    LOG = pd.concat(logs, ignore_index=True) if logs else pd.DataFrame()
    FIN = pd.DataFrame(fins)
    if len(FIN) and "lambda_max_hessiano" not in FIN:
        FIN["lambda_max_hessiano"] = np.nan
    return LOG, FIN


LOG, FIN = cargar_resultados()
print(f"{LOG['id'].nunique() if len(LOG) else 0} corridas con log | {len(FIN)} terminadas")


def ic95(x):
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return float("nan")
    from scipy import stats
    return stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)


def media_ic(serie):
    m, h = np.mean(serie), ic95(serie)
    return f"{m:.2f}" if np.isnan(h) else f"{m:.2f} ± {h:.2f}"


# filtros que se usan en las figuras
def teoria(df):
    # corridas sin momentum y con el numero completo de epocas
    return df[(df.momentum == cfg.BASE["momentum"]) & (df.epocas == cfg.P()["epocas"])]


def replica(df):
    return df[(df.lam == 0) & (df.momentum == cfg.BASE["momentum_R"]) & (df.metodo == "sam")
              & (df.rho_sched == "const") & (df.epocas == cfg.P()["epocas"])]

# %% tiempos
if len(LOG):
    t = LOG.groupby("id").agg(epocas_hechas=("epoca", "max"), seg_por_epoca=("seg_epoca", "median"),
                              horas=("seg_epoca", lambda s: s.sum() / 3600))
    mostrar(t)
    print(f"total: {t.horas.sum():.2f} h de entrenamiento\n")

# %% L: piloto de lambda
# para cada lambda del piloto, la densidad y la precision de Prox-SGD; el mas grande que deje
# una fraccion clara de pesos en cero sin botar la precision va en lam_ref (config.py)

if len(FIN):
    Lp = FIN[(FIN.grupo == "L")]
    if len(Lp):
        mostrar(Lp.groupby(["dataset", "lam"]).agg(densidad=("densidad_final", "mean"),
                val=("acc_val_final", "mean"), test=("acc_test_final", "mean")))
    else:
        print("Aún no hay corridas del piloto.")


# %% R: replica de la Tabla 4 del paper (ResNet-18, 200 epocas)
REF9 = {"cifar10": {"const": 94.10, "dim1": 93.95, "dim2": 94.60, "dim3": 94.75},
        "cifar100": {"const": 71.77, "dim1": 74.43, "dim2": 73.40, "dim3": 75.65}}
NOMBRE_SCHED = {"const": "Constante", "dim1": "Diminish 1", "dim2": "Diminish 2", "dim3": "Diminish 3"}

if len(FIN):
    R = replica(FIN)
    filas = []
    for ds in cfg.DATASETS:
        for sch in ["const", "dim1", "dim2", "dim3"]:
            sub = R[(R.dataset == ds) & (R.lr_sched == sch)]
            if sch == "const" and len(sub):
                # igual que en el paper: el mejor paso constante de la grilla
                mejor = sub.groupby("eta1").acc_val_mejor.mean().idxmax()
                sub = sub[sub.eta1 == mejor]
            if len(sub):
                filas.append(dict(dataset=ds, paso=NOMBRE_SCHED[sch], **{"test final (%)": media_ic(sub.acc_test_final),
                                  "test en mejor val (%)": media_ic(sub.acc_test_mejor_val),
                                  "paper (%)": REF9[ds][sch], "semillas": len(sub)}))
    mostrar(pd.DataFrame(filas)) if filas else print("Aún no hay corridas terminadas del grupo R.")

def figura_R():
    if not len(LOG):
        return
    R = replica(LOG)
    if not len(R):
        print("Sin datos del grupo R.")
        return
    cols = {"const": GRIS, "dim1": C1, "dim2": C2, "dim3": C3}
    fig, axs = plt.subplots(2, len(cfg.DATASETS), figsize=(5.2 * len(cfg.DATASETS), 6.4), squeeze=False)
    for j, ds in enumerate(cfg.DATASETS):
        for sch in ["const", "dim1", "dim2", "dim3"]:
            sub = R[(R.dataset == ds) & (R.lr_sched == sch)]
            if not len(sub):
                continue
            if sch == "const":
                sub = sub[sub.eta1 == sub.groupby("eta1").acc_val.max().idxmax()]
            g = sub.groupby("epoca")
            axs[0, j].semilogy(g.perdida_train.mean(), color=cols[sch], label=NOMBRE_SCHED[sch])
            axs[1, j].plot(g.acc_test.mean(), color=cols[sch], label=NOMBRE_SCHED[sch])
        axs[0, j].set_title(f"{ds.upper()}: pérdida de entrenamiento", loc="left")
        axs[1, j].set_title(f"{ds.upper()}: precisión de prueba (%)", loc="left")
        for a in axs[:, j]:
            a.set_xlabel("época")
            a.grid(True)
    axs[0, 0].legend(frameon=False)
    fig.tight_layout()
    guardar(fig, "R_replica")
    plt.show()


figura_R()


# %% P y S: radio constante, reglas de paso y recorte
# lineas llenas: Prox-SAM con radio constante; punteadas: Prox-SGD; violeta: Prox-SAM recortado
def figura_P():
    if not len(LOG):
        return
    T = teoria(LOG)
    T = T[(T.lam == cfg.BASE["lam_ref"])]
    cols = {"const": GRIS, "dim1": C1, "dim2": C2, "dim3": C3}
    for ds in cfg.DATASETS:
        D = T[T.dataset == ds]
        if not len(D[(D.metodo == "sam") & (D.rho_sched == "const")]):
            print(f"Sin datos de P para {ds}.")
            continue
        fig, axs = plt.subplots(2, 3, figsize=(14, 7.4))
        curvas = [("sam", "const", sch, cols[sch], "-", f"Prox-SAM, {NOMBRE_SCHED[sch]}") for sch in cols]
        curvas += [("sgd", "cero", sch, cols[sch], ":", f"Prox-SGD, {NOMBRE_SCHED[sch]}") for sch in ["const", "dim2"]]
        curvas += [("sam_rec", "const", "dim2", C7, "-", "Prox-SAM recortado, Diminish 2")]
        for met, rs, sch, col, ls, lab in curvas:
            sub = D[(D.metodo == met) & (D.rho_sched == rs) & (D.lr_sched == sch)]
            if sch == "const":
                sub = sub[sub.eta1 == cfg.BASE["eta_const"]]
            if not len(sub):
                continue
            g = sub.groupby("epoca")
            gc = sub.dropna(subset=["certificado"]).groupby("epoca")
            axs[0, 0].semilogy(gc.certificado.mean(), color=col, ls=ls, marker="o", ms=3, label=lab)
            axs[0, 1].semilogy(gc.dist_fuerte.mean(), color=col, ls=ls, marker="o", ms=3)
            axs[0, 2].plot(100 * g.densidad.mean(), color=col, ls=ls)
            axs[1, 0].plot(g.acc_test.mean(), color=col, ls=ls)
            if met != "sgd":
                axs[1, 1].semilogy(g.rho_ef.mean(), color=col, ls=ls)
                axs[1, 2].semilogy(g.L_sec.mean(), color=col, ls=ls)
        tit = [r"(a) $\|\mathcal{G}_{\bar t}(w)\|$", r"(b) $\mathrm{dist}(0,\partial F(w))$", "(c) Pesos no nulos (%)",
               "(d) Precisión de prueba (%)", r"(e) Radio efectivo $\|y^k-w^k\|$", r"(f) Estimación de $L$ ($\|g^k-g_1\|/\|y^k-w^k\|$)"]
        for a, t in zip(axs.flat, tit):
            a.set_title(t, loc="left")
            a.set_xlabel("época")
            a.grid(True)
        axs[0, 0].legend(frameon=False, fontsize=7)
        fig.suptitle(f"{ds.upper()}, $\\lambda$ = {cfg.BASE['lam_ref']:g}", x=0.01, ha="left", fontsize=10)
        fig.tight_layout()
        guardar(fig, f"P_radio_constante_{ds}")
        plt.show()
    if len(FIN):
        t = teoria(FIN)
        t = t[(t.lam == cfg.BASE["lam_ref"]) & t.metodo.isin(["sgd", "sam", "sam_rec"]) & t.rho_sched.isin(["cero", "const"])]
        if len(t):
            fin_cert = LOG.dropna(subset=["certificado"]).sort_values("epoca").groupby("id").tail(1)[["id", "certificado", "dist_fuerte"]]
            t = t.merge(fin_cert, on="id", how="left")
            mostrar(t.groupby(["dataset", "metodo", "lr_sched"]).agg(G_final=("certificado", "mean"),
                    dist_final=("dist_fuerte", "mean"), densidad=("densidad_final", "mean"),
                    test_final=("acc_test_final", media_ic), semillas=("semilla", "count")))


figura_P()


# %% N1: radios con Diminish 2
def figura_N1():
    if not len(LOG):
        return
    T = teoria(LOG)
    N1 = T[(T.lam == cfg.BASE["lam_ref"]) & (T.metodo == "sam") & (T.lr_sched == "dim2")]
    if N1.rho_sched.nunique() < 2:
        print("Sin datos suficientes de N1.")
        return
    cols = {"const": GRIS, "hip31": C3, "paper9": C1}
    etq = {"const": r"$\rho$ constante", "hip31": r"$\rho_n\propto n^{-0.26}$ (solo $\sum t_n\rho_n^2<\infty$)",
           "paper9": r"$\rho_n\propto n^{-0.51}$ (también $\sum t_n\rho_n<\infty$)"}
    fig, axs = plt.subplots(1, 4, figsize=(16, 3.8))
    for ds, ls in zip(cfg.DATASETS, ["-", "--"]):
        for rs in ["const", "hip31", "paper9"]:
            sub = N1[(N1.dataset == ds) & (N1.rho_sched == rs)]
            if not len(sub):
                continue
            g = sub.groupby("epoca")
            lab = f"{etq[rs]}, {ds}" if len(cfg.DATASETS) > 1 else etq[rs]
            axs[0].plot(g.rho.mean(), color=cols[rs], ls=ls, label=lab)
            axs[1].plot(g.acc_test.mean(), color=cols[rs], ls=ls)
            gc = sub.dropna(subset=["certificado"]).groupby("epoca")
            axs[2].semilogy(gc.certificado.mean(), color=cols[rs], ls=ls, marker="o", ms=3)
            axs[3].semilogy(gc.dist_fuerte.mean(), color=cols[rs], ls=ls, marker="o", ms=3)
    axs[0].set_title(r"(a) Radio $\rho_n$", loc="left")
    axs[1].set_title("(b) Precisión de prueba (%)", loc="left")
    axs[2].set_title(r"(c) $\|\mathcal{G}_{\bar t}(w)\|$", loc="left")
    axs[3].set_title(r"(d) $\mathrm{dist}(0,\partial F(w))$", loc="left")
    for a in axs:
        a.set_xlabel("época")
        a.grid(True)
    axs[0].legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    guardar(fig, "N1_radios")
    plt.show()
    if len(FIN):
        t = teoria(FIN)
        t = t[(t.lam == cfg.BASE["lam_ref"]) & (t.metodo == "sam") & (t.lr_sched == "dim2")]
        mostrar(t.groupby(["dataset", "rho_sched"]).agg(test_final=("acc_test_final", media_ic),
                densidad=("densidad_final", "mean"), lambda_max=("lambda_max_hessiano", "mean"),
                semillas=("semilla", "count")))


figura_N1()


# %% N2: entrenamiento disperso (solo si se corrio el grupo N2)
NOM_MET = {"sgd": "Prox-SGD", "sam": "Prox-SAM", "sam_map": "Prox-SAM-map", "usam": "Prox-USAM", "usam_corr": "Prox-USAM corr."}
COL_MET = {"sgd": GRIS, "sam": C1, "sam_map": C2, "usam": C8, "usam_corr": C3}
MARK_LAM = ["o", "s", "D", "^", "v"]


def figura_N2():
    if not len(FIN):
        print("Sin corridas terminadas.")
        return
    N2 = teoria(FIN)
    N2 = N2[(N2.lr_sched == "dim2") & N2.metodo.isin(["sgd", "sam", "sam_map"])
            & N2.rho_sched.isin(["cero", "hip31"])]
    if not len(N2):
        print("Sin datos de N2.")
        return
    lams = sorted(N2.lam.unique())
    fig = plt.figure(figsize=(13, 8.6))
    gs = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.36)
    # (a) frontera precision-densidad
    ax = fig.add_subplot(gs[0, 0])
    for ds, ls in zip(cfg.DATASETS, ["-", "--"]):
        for met in ["sgd", "sam", "sam_map"]:
            sub = N2[(N2.dataset == ds) & (N2.metodo == met)].groupby("lam").agg(
                dens=("densidad_final", "mean"), acc=("acc_test_final", "mean")).reset_index()
            if not len(sub):
                continue
            ax.plot(100 * sub.dens, sub.acc, color=COL_MET[met], ls=ls, lw=1.0, alpha=0.7)
            for _, r in sub.iterrows():
                ax.scatter(100 * r.dens, r.acc, color=COL_MET[met], marker=MARK_LAM[lams.index(r.lam) % 5], s=40,
                           edgecolor="white", zorder=4)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=COL_MET[m], lw=2, label=NOM_MET[m]) for m in ["sgd", "sam", "sam_map"]]
    h += [Line2D([], [], ls="", marker=MARK_LAM[i], color=GRIS, label=fr"$\lambda={l:g}$") for i, l in enumerate(lams)]
    if len(cfg.DATASETS) > 1:
        h += [Line2D([], [], color=TINTA2, ls=ls, label=ds) for ds, ls in zip(cfg.DATASETS, ["-", "--"])]
    ax.legend(handles=h, frameon=False, fontsize=7)
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.set_xlabel("pesos no nulos (%)")
    ax.set_ylabel("precisión de prueba (%)")
    ax.set_title("(a) Frontera precisión–densidad", loc="left")
    ax.grid(True)
    # (b) robustez a poda adicional
    ax = fig.add_subplot(gs[0, 1])
    ds0, lam0 = cfg.DATASETS[0], lams[-1]
    for met in ["sgd", "sam", "sam_map"]:
        sub = N2[(N2.dataset == ds0) & (N2.metodo == met) & (N2.lam == lam0)]
        if not len(sub) or "robustez_poda" not in sub:
            continue
        curvas = [pd.Series({float(k): v for k, v in rp.items()}) for rp in sub.robustez_poda if isinstance(rp, dict)]
        if not curvas:
            continue
        cc = pd.concat(curvas, axis=1).mean(axis=1).sort_index()
        ax.plot(100 * cc.index, cc.values, color=COL_MET[met], marker="o", ms=4, label=NOM_MET[met])
    ax.set_xlabel("fracción podada por magnitud (%)")
    ax.set_ylabel("precisión de prueba (%)")
    ax.set_title(fr"(b) Robustez a poda adicional ({ds0}, $\lambda={lam0:g}$)", loc="left")
    ax.grid(True)
    ax.legend(frameon=False)
    # (c) agudeza
    ax = fig.add_subplot(gs[0, 2])
    if "lambda_max_hessiano" in N2:
        piv = N2.groupby(["metodo", "lam"]).lambda_max_hessiano.mean().unstack(0)
        x = np.arange(len(piv.index))
        w = 0.8 / max(len(piv.columns), 1)
        for i, met in enumerate([m for m in ["sgd", "sam", "sam_map"] if m in piv.columns]):
            ax.bar(x + (i - 1) * w, piv[met].values, width=w * 0.92, color=COL_MET[met], label=NOM_MET[met])
        ax.set_xticks(x)
        ax.set_xticklabels([fr"$\lambda={l:g}$" for l in piv.index])
        ax.set_ylabel(r"$\lambda_{\max}(\nabla^2 f)$")
        ax.legend(frameon=False, fontsize=7.5)
    ax.set_title("(c) Agudeza al final del entrenamiento", loc="left")
    # (d) densidad por capa (mapa de calor)
    ax = fig.add_subplot(gs[1, :2])
    filas, nombres = [], []
    for met in ["sgd", "sam", "sam_map"]:
        sub = N2[(N2.dataset == ds0) & (N2.metodo == met) & (N2.lam == lam0)]
        if not len(sub):
            continue
        dpc = pd.DataFrame([r for r in sub.densidad_por_capa]).mean()
        filas.append(dpc.values)
        nombres.append(NOM_MET[met])
        capas = list(dpc.index)
    if filas:
        Z = np.vstack(filas)
        im = ax.imshow(Z, aspect="auto", cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
        ax.set_yticks(range(len(nombres)))
        ax.set_yticklabels(nombres)
        ax.set_xticks(range(len(capas)))
        ax.set_xticklabels([c.replace(".weight", "").replace("capas.", "") for c in capas],
                                                            rotation=90, fontsize=6.5)
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    ax.set_title(fr"(d) Densidad por capa ({ds0}, $\lambda={lam0:g}$): de la entrada (izq.) a la salida (der.)", loc="left")
    # (e) certificado vs norma del gradiente
    ax = fig.add_subplot(gs[1, 2])
    L2 = teoria(LOG)
    L2 = L2[(L2.lr_sched == "dim2") & L2.rho_sched.isin(["cero", "hip31"]) & (L2.dataset == ds0)
            & (L2.lam == lam0)].dropna(subset=["certificado"])
    for met in ["sgd", "sam", "sam_map"]:
        sub = L2[L2.metodo == met].groupby("epoca")
        if not len(L2[L2.metodo == met]):
            continue
        ax.semilogy(sub.certificado.mean(), color=COL_MET[met], marker="o", ms=3, label=NOM_MET[met])
        ax.semilogy(sub.norma_grad.mean(), color=COL_MET[met], ls=":", lw=1.2)
    ax.set_xlabel("época")
    ax.set_title(r"(e) $\|\mathcal{G}_{\bar t}\|$ (sólida) y $\|\nabla f\|$ (punteada)", loc="left")
    ax.grid(True)
    ax.legend(frameon=False, fontsize=7.5)
    guardar(fig, "N2_disperso")
    plt.show()
    mostrar(N2.groupby(["dataset", "lam", "metodo"]).agg(
        test_final=("acc_test_final", media_ic), densidad=("densidad_final", "mean"),
        lambda_max=("lambda_max_hessiano", "mean"), semillas=("semilla", "count")))


figura_N2()


# %% N3: Prox-USAM contra el corregido (solo si se corrio el grupo N3)
def figura_N3():
    if not len(LOG):
        return
    N3 = LOG[LOG.metodo.isin(["usam", "usam_corr"])]
    if not len(N3):
        print("Sin datos de N3.")
        return
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    for met in ["usam", "usam_corr"]:
        for ds, ls in zip(cfg.DATASETS, ["-", "--"]):
            sub = N3[(N3.metodo == met) & (N3.dataset == ds)].groupby("epoca")
            if not len(N3[(N3.metodo == met) & (N3.dataset == ds)]):
                continue
            axs[0].plot(100 * sub.densidad.mean(), color=COL_MET[met], ls=ls, label=f"{NOM_MET[met]}, {ds}")
            axs[1].plot(sub.acc_test.mean(), color=COL_MET[met], ls=ls)
    axs[0].set_title("(a) Pesos no nulos (%)", loc="left")
    axs[1].set_title("(b) Precisión de prueba (%)", loc="left")
    for a in axs:
        a.set_xlabel("época")
        a.grid(True)
    axs[0].legend(frameon=False)
    fig.tight_layout()
    guardar(fig, "N3_usam")
    plt.show()


figura_N3()
