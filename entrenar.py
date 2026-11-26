# Cola de experimentos y entrenamiento de una corrida, con guardado parcial
import os, csv, json, time
import numpy as np
import torch
import config as cfg
from datos import datos_para, aumentar
from modelo import crear_modelo
from optimizador import ProxSAM
from metricas import paso_lr, radio, evaluar, densidad, densidad_por_capa, certificado

CAMPOS = ["epoca", "lr", "rho", "lam", "perdida_train", "acc_train", "acc_val", "perd_val", "acc_test",
          "densidad", "certificado", "dist_fuerte", "norma_grad", "rho_ef", "L_sec", "L_hat",
          "escala", "descartados", "seg_epoca"]
CAMPOS_IT = ["epoca", "iter", "perdida", "lr", "rho", "L_hat", "seg_iter"]


class Detener(Exception):
    # se lanza cuando hay que parar (limite de horas o archivo PARAR.txt); el avance ya quedo guardado
    pass


class Control:
    # decide cuando hay que parar
    def __init__(self, max_horas=None, max_iters=None):
        self.t0 = time.time()
        self.limite = self.t0 + 3600 * max_horas if max_horas else None
        self.max_iters, self.iters = max_iters, 0

    def hay_que_parar(self):
        self.iters += 1
        if self.max_iters and self.iters >= self.max_iters:
            return "se alcanzo el numero maximo de iteraciones"
        if self.limite and time.time() > self.limite:
            return "se alcanzo el limite de horas"
        if self.iters % 20 == 0 and os.path.exists(os.path.join(cfg.RAIZ, "PARAR.txt")):
            return "existe el archivo PARAR.txt"
        return None


def registrar(msg):
    # imprime y deja copia en resultados/registro.txt
    print(msg, flush=True)
    os.makedirs(cfg.RAIZ, exist_ok=True)
    with open(os.path.join(cfg.RAIZ, "registro.txt"), "a", encoding="utf-8") as fh:
        fh.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")


def id_corrida(c):
    # el grupo no va en el nombre: una corrida que sirve a dos grupos se hace una sola vez
    return (f"{c['dataset']}_{c['modelo']}_{c['metodo']}_lr-{c['lr_sched']}"
            f"{'' if c['lr_sched'] != 'const' else '-' + str(c['eta1'])}_rho-{c['rho_sched']}-{c['rho0']}"
            f"_lam-{c['lam']}_mom-{c['momentum']}_ep{c['epocas']}_s{c['semilla']}")


def cola_experimentos():
    B, P = cfg.BASE, cfg.P()
    cola = []

    def add(grupo, ds, s, **kw):
        c = dict(grupo=grupo, dataset=ds, semilla=s, modelo=B["modelo"], batch=B["batch"],
                 momentum=B["momentum"], wd=B["wd"], eta1=B["eta1"], rho0=B["rho0"],
                 epocas=P["epocas"], lam=B["lam_ref"], metodo="sam", lr_sched="dim2", rho_sched="const")
        c.update(kw)
        cola.append(c)

    sgd = dict(metodo="sgd", rho_sched="cero", rho0=0.0)

    def con_regla(sch, eta_const):
        return dict(lr_sched="const", eta1=eta_const) if sch == "const" else dict(lr_sched=sch)

    for ds in cfg.DATASETS:
        for s in P["semillas"]:
            if "L" in cfg.GRUPOS and s == P["semillas"][0]:
                for lam in B["lam_piloto"]:
                    add("L", ds, s, lam=lam, epocas=min(P["epocas"], B["epocas_piloto"]), **sgd)
            if "R" in cfg.GRUPOS:
                kw = dict(lam=0.0, momentum=B["momentum_R"])
                for sch in B["reglas_R"]:
                    if sch == "const":
                        for eta in (B["etas_R_completo"] if cfg.PRESET == "completo" else B["etas_R"]):
                            add("R", ds, s, lr_sched="const", eta1=eta, **kw)
                    else:
                        add("R", ds, s, lr_sched=sch, **kw)
            if "P" in cfg.GRUPOS:
                for sch in B["reglas_P"]:
                    add("P", ds, s, **con_regla(sch, B["eta_const"]))
                for sch in B["reglas_P_sgd"]:
                    add("P", ds, s, **con_regla(sch, B["eta_const"]), **sgd)
            if "N1" in cfg.GRUPOS:
                for rs in ["const"] + list(B["radios_N1"]):
                    add("N1", ds, s, rho_sched=rs)
            if "S" in cfg.GRUPOS:
                add("S", ds, s, metodo="sam_rec")
                add("S", ds, s)
                add("S", ds, s, **sgd)
            if "N2" in cfg.GRUPOS:
                for lam in B["lam_grid"]:
                    add("N2", ds, s, lam=lam, **sgd)
                    add("N2", ds, s, lam=lam, rho_sched="hip31")
                    add("N2", ds, s, metodo="sam_map", lam=lam, rho_sched="hip31")
            if "N3" in cfg.GRUPOS:
                for met in ["usam", "usam_corr"]:
                    add("N3", ds, s, metodo=met, rho0=B["rho_usam"])
    vistos, unica = set(), []
    for c in cola:
        k = id_corrida(c)
        if k not in vistos:
            vistos.add(k)
            unica.append(c)
    return unica


def estado_corrida(c):
    # devuelve (estado, epocas hechas, precision de prueba)
    carpeta = os.path.join(cfg.RAIZ, id_corrida(c))
    if os.path.exists(os.path.join(carpeta, "final.json")):
        fin = json.load(open(os.path.join(carpeta, "final.json")))
        return "terminada", c["epocas"], fin["acc_test_final"]
    f_log = os.path.join(carpeta, "log.csv")
    if os.path.exists(f_log):
        filas = list(csv.DictReader(open(f_log)))
        if filas:
            return "a medias", int(filas[-1]["epoca"]), float(filas[-1]["acc_test"])
    if os.path.exists(os.path.join(carpeta, "ckpt.pt")):
        return "a medias", 0, float("nan")
    return "pendiente", 0, float("nan")


def escribir_estado(cola):
    # resultados/estado.csv: una linea por corrida de la cola
    os.makedirs(cfg.RAIZ, exist_ok=True)
    with open(os.path.join(cfg.RAIZ, "estado.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["grupo", "id", "estado", "epocas_hechas", "epocas", "acc_test"])
        for c in cola:
            est, ep, acc = estado_corrida(c)
            w.writerow([c["grupo"], id_corrida(c), est, ep, c["epocas"], f"{acc:.2f}"])


def escribir_resumen():
    # resultados/resumen.csv: una linea por corrida terminada
    import glob
    filas = []
    for f in sorted(glob.glob(os.path.join(cfg.RAIZ, "*", "final.json"))):
        fin = json.load(open(f))
        fila = {**fin["config"], **{k: v for k, v in fin.items() if not isinstance(v, (dict, list))}}
        filas.append(fila)
    if not filas:
        return
    campos = sorted({k for f in filas for k in f})
    with open(os.path.join(cfg.RAIZ, "resumen.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=campos)
        w.writeheader()
        w.writerows(filas)


def _guardar(ruta, modelo, opt, epoca, filas, parcial=None):
    torch.save(dict(modelo=modelo.state_dict(), opt=opt.estado(), epoca=epoca, filas=filas, parcial=parcial,
                    rng_cpu=torch.get_rng_state(),
                    rng_cuda=torch.cuda.get_rng_state() if cfg.DEVICE == "cuda" else None), ruta + ".tmp")
    os.replace(ruta + ".tmp", ruta)


def entrenar(c, control=None):
    # entrena una corrida. Guarda al final de cada epoca y, dentro de la epoca, cada BASE["min_guardado"] minutos
    B = cfg.BASE
    control = control or Control()
    rid = id_corrida(c)
    carpeta = os.path.join(cfg.RAIZ, rid)
    os.makedirs(carpeta, exist_ok=True)
    if os.path.exists(os.path.join(carpeta, "final.json")):
        registrar(f"[ya terminada] {rid}")
        return json.load(open(os.path.join(carpeta, "final.json")))
    json.dump(dict(config=c, base=B, preset=cfg.PRESET, equipo=cfg.info_equipo()),
              open(os.path.join(carpeta, "config.json"), "w"), indent=1)
    d = datos_para(c["dataset"])
    torch.manual_seed(c["semilla"])
    np.random.seed(c["semilla"])
    modelo = crear_modelo(c["modelo"], d["n_clases"])
    opt = ProxSAM(modelo, c["metodo"], c["momentum"], c["wd"], B["nu_rec"], B["L0"])
    ep0, filas, parcial = 1, [], None
    ruta_ckpt = os.path.join(carpeta, "ckpt.pt")
    st = None
    if os.path.exists(ruta_ckpt):
        st = torch.load(ruta_ckpt, map_location=cfg.DEVICE, weights_only=False)
        if not all(bool(torch.isfinite(v).all()) for v in st["modelo"].values() if v.is_floating_point()):
            registrar(f"[checkpoint con nan, se descarta] {rid}")
            st = None
            os.remove(ruta_ckpt)
    if st is not None:
        modelo.load_state_dict(st["modelo"])
        opt.cargar(st["opt"])
        ep0, filas, parcial = st["epoca"] + 1, st["filas"], st.get("parcial")
        torch.set_rng_state(st["rng_cpu"])
        if cfg.DEVICE == "cuda":
            torch.cuda.set_rng_state(st["rng_cuda"])
        donde = f"época {ep0}" + (f", iteración {parcial['i'] // c['batch']}" if parcial else "")
        registrar(f"[reanudando] {rid} desde la {donde}")
    else:
        registrar(f"[nueva] {rid}")
    Xc = d["Xtr"][: B["n_certificado"], :, 4:36, 4:36]
    Yc = d["Ytr"][: B["n_certificado"]]
    N, Bt = len(d["Xtr"]), c["batch"]
    n_its = (N + Bt - 1) // Bt
    ruta_it = os.path.join(carpeta, "iter_log.csv")
    if not os.path.exists(ruta_it):
        with open(ruta_it, "w", newline="") as fh:
            csv.writer(fh).writerow(CAMPOS_IT)
    for ep in range(ep0, c["epocas"] + 1):
        lr, rho = paso_lr(c["lr_sched"], ep, c["eta1"]), radio(c["rho_sched"], ep, c["rho0"], B["delta"])
        if c["metodo"] in ("usam", "usam_corr"):
            rho = c["rho0"]
        if parcial and parcial["epoca"] == ep:
            # se retoma la epoca donde quedo
            perm, i0 = parcial["perm"].to(cfg.DEVICE), parcial["i"]
            suma_perd = torch.tensor(parcial["suma_perd"], device=cfg.DEVICE)
            suma_ok = torch.tensor(parcial["suma_ok"], device=cfg.DEVICE)
            n_vistos, t_previo = parcial["n_vistos"], parcial["seg"]
        else:
            opt.reiniciar_stats()
            perm, i0 = torch.randperm(N, device=cfg.DEVICE), 0
            suma_perd = torch.zeros((), device=cfg.DEVICE)
            suma_ok = torch.zeros((), device=cfg.DEVICE)
            n_vistos, t_previo = 0, 0.0
        parcial = None
        t0 = t_guardado = t_print = time.time()
        modelo.train()

        def guardar_parcial(i_sig):
            _guardar(ruta_ckpt, modelo, opt, ep - 1, filas,
                     dict(epoca=ep, i=i_sig, perm=perm.cpu(), suma_perd=suma_perd.item(), suma_ok=suma_ok.item(),
                          n_vistos=n_vistos, seg=t_previo + time.time() - t0))

        for i in range(i0, N, Bt):
            idx = perm[i:i + Bt]
            x = aumentar(d["Xtr"][idx], d, B["cutout"], B["cutout_p"])
            y = d["Ytr"][idx]
            perd, out, valido = opt.paso(x, y, lr, rho, c["lam"], B["tbar"])
            if valido:
                suma_perd += perd.float() * len(idx)
                suma_ok += (out.argmax(1) == y).sum()
                n_vistos += len(idx)
            it = i // Bt + 1
            ahora = time.time()
            if it % B["cada_print"] == 0 or it == n_its:
                seg_it = (ahora - t_print) / (B["cada_print"] if it % B["cada_print"] == 0 else max(it % B["cada_print"], 1))
                t_print = ahora
                pm = (suma_perd / max(n_vistos, 1)).item()
                print(f"    ep {ep}/{c['epocas']} it {it}/{n_its}  perd={pm:.3f}  {seg_it:.2f} s/it  "
                      f"faltan {seg_it * (n_its - it) / 60:.1f} min de esta época", flush=True)
                with open(ruta_it, "a", newline="") as fh:
                    csv.writer(fh).writerow([ep, it, f"{perd.item():.5f}", lr, rho, f"{opt.L_hat:.4f}", f"{seg_it:.4f}"])
            motivo = control.hay_que_parar()
            if motivo and it < n_its:
                guardar_parcial(i + Bt)
                registrar(f"  [detenido] {motivo}; avance guardado en la época {ep}, iteración {it}")
                raise Detener(motivo)
            if ahora - t_guardado > 60 * B["min_guardado"] and it < n_its:
                guardar_parcial(i + Bt)
                t_guardado = time.time()
        if cfg.DEVICE == "cuda":
            torch.cuda.synchronize()
        seg_train = t_previo + time.time() - t0
        acc_val, perd_val = evaluar(modelo, d["Xva"], d["Yva"], d)
        acc_test, _ = evaluar(modelo, d["Xte"], d["Yte"], d)
        cert = ngr = dfu = float("nan")
        if ep % B["cada_certificado"] == 0 or ep == c["epocas"] or ep == 1:
            cert, ngr, dfu = certificado(modelo, Xc, Yc, d, c["lam"], B["tbar"], c["wd"])
        ns = max(opt.n_stats, 1)
        fila = dict(epoca=ep, lr=lr, rho=rho, lam=c["lam"],
                    perdida_train=(suma_perd / max(n_vistos, 1)).item(),
                    acc_train=100 * (suma_ok / max(n_vistos, 1)).item(),
                    acc_val=acc_val, perd_val=perd_val, acc_test=acc_test, densidad=densidad(modelo),
                    certificado=cert, dist_fuerte=dfu, norma_grad=ngr,
                    rho_ef=opt.s_rho / ns if opt.n_stats else 0.0,
                    L_sec=opt.s_sec / ns if opt.n_stats else float("nan"), L_hat=opt.L_hat, escala=opt.S,
                    descartados=opt.descartados, seg_epoca=seg_train)
        filas.append(fila)
        with open(os.path.join(carpeta, "log.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=CAMPOS)
            w.writeheader()
            w.writerows(filas)
        _guardar(ruta_ckpt, modelo, opt, ep, filas)
        registrar(f"  ep {ep:3d}/{c['epocas']}  lr={lr:.4f} rho={rho:.4f}  perd={fila['perdida_train']:.3f}  "
                  f"val={acc_val:.2f}%  test={acc_test:.2f}%  dens={fila['densidad']:.3f}  "
                  f"cert={cert:.3g}  dist={dfu:.3g}  {seg_train:.0f} s/época")
        motivo = control.hay_que_parar() if ep < c["epocas"] else None
        if motivo:
            registrar(f"  [detenido] {motivo}; avance guardado al terminar la época {ep}")
            raise Detener(motivo)
    # metricas finales
    best = max(filas, key=lambda r: r["acc_val"])
    fin = dict(id=rid, config=c, acc_test_final=filas[-1]["acc_test"], acc_test_mejor_val=best["acc_test"],
               acc_val_final=filas[-1]["acc_val"], acc_val_mejor=best["acc_val"],
               epoca_mejor_val=best["epoca"], perdida_train_final=filas[-1]["perdida_train"],
               certificado_final=filas[-1]["certificado"], dist_fuerte_final=filas[-1]["dist_fuerte"],
               norma_grad_final=filas[-1]["norma_grad"], densidad_final=filas[-1]["densidad"],
               L_sec_final=filas[-1]["L_sec"], rho_ef_final=filas[-1]["rho_ef"], descartados=filas[-1]["descartados"],
               densidad_por_capa=densidad_por_capa(modelo),
               seg_por_epoca=float(np.median([r["seg_epoca"] for r in filas])),
               horas_total=float(np.sum([r["seg_epoca"] for r in filas]) / 3600))
    json.dump(fin, open(os.path.join(carpeta, "final.json"), "w"), indent=1)
    registrar(f"  -> test final {fin['acc_test_final']:.2f}% | test en mejor val {fin['acc_test_mejor_val']:.2f}% | "
              f"densidad {fin['densidad_final']:.3f} | {fin['horas_total']:.2f} h")
    return fin


def medir_seg_por_epoca(metodo="sam", pasos=30):
    # mide cuanto se demora una epoca (sin evaluacion) con unos pocos pasos
    B = cfg.BASE
    d = datos_para(cfg.DATASETS[0])
    m = crear_modelo(B["modelo"], d["n_clases"])
    opt = ProxSAM(m, metodo, 0.0, B["wd"])
    N, Bt = len(d["Xtr"]), B["batch"]
    for k in range(pasos + 3):
        if k == 3:
            if cfg.DEVICE == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
        idx = torch.randint(0, N, (Bt,), device=cfg.DEVICE)
        opt.paso(aumentar(d["Xtr"][idx], d), d["Ytr"][idx], 0.05, 0.05, 1e-5, 0.1)
    if cfg.DEVICE == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t0) / pasos * ((N + Bt - 1) // Bt)


def estimar_horas(cola, pasos=30):
    s_sam, s_sgd = medir_seg_por_epoca("sam", pasos), medir_seg_por_epoca("sgd", pasos)
    total = 0.0
    for c in cola:
        est, ep, _ = estado_corrida(c)
        if est != "terminada":
            total += (s_sgd if c["metodo"] == "sgd" else s_sam) * (c["epocas"] - ep)
    return s_sam, s_sgd, total / 3600
