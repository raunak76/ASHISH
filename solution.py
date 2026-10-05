"""Reference shelf matching: neural pair scorer (PyTorch, GPU when available).

Usage: python3 solution.py <public_dir> <submission_out>

A trained neural network scores every (entry, card) pair of a case for the
lead and companion roles and predicts the description tier from the
(entry, lead card) representation. CPU code only parses data, builds input
features and enforces card capacity via a linear assignment over the
network's log-probabilities.
"""
import sys, os, json, re, math, random, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

TIERS = ["long_grammar", "grammar", "grammar_sketch", "phonology_or_text", "wordlist_or_less"]
NB = 1 << 18
SEED = 1234
VAL = os.environ.get("VAL", "0") == "1"
EPOCHS = int(os.environ.get("EPOCHS", "8"))
N_MODELS = int(os.environ.get("N_MODELS", "3"))
CTX_LAYERS = int(os.environ.get("CTX_LAYERS", "2"))
USE_GOLD_STATS = os.environ.get("GOLDSTATS", "1") == "1"
SELF_TRAIN = int(os.environ.get("SELF_TRAIN", "1"))
CONF = float(os.environ.get("CONF", "0.5"))
PSEUDO = int(os.environ.get("PSEUDO", "0"))
TRI = os.environ.get("TRI", "1") == "1"
EMBW = float(os.environ.get("EMBW", "1"))
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TOK = re.compile(r"\w+", re.UNICODE)
STOP = set("a an the of and in on to for with by de la le les des du et en der die das und von zur zum im el los las y del da do dos e i o".split())


def toks(s):
    s = (s or "").replace("[ENTRY]", " xentryx ")
    return [t for t in TOK.findall(s.lower())]


def ctoks(s):
    # creator surnames (token before a comma) plus all tokens
    out = set()
    for part in re.split(r"\band\b|;", s or ""):
        part = part.strip()
        if not part:
            continue
        sur = part.split(",")[0].strip().lower()
        if sur:
            out.add(sur)
    return out


def years(s):
    return [int(y) for y in re.findall(r"(1[5-9]\d\d|20\d\d)", s or "")]


def h(t):
    return int(hashlib.md5(t.encode("utf8")).hexdigest()[:8], 16) % (NB - 1) + 1


def tri(s):
    s = f"#{s}#"
    return {s[i:i + 3] for i in range(len(s) - 2)}


KW = ["grammar", "grammatik", "grammaire", "gramática", "grammatica", "sketch", "outline", "esquisse", "phonology",
      "phonologie", "phonetics", "dictionary", "dictionnaire", "wörterbuch", "diccionario", "vocabulary", "vocabulaire",
      "wordlist", "lexicon", "texts", "textes", "morphology", "syntax", "verb", "notes", "description", "descriptive",
      "introduction", "survey", "comparative", "language", "dialect", "study", "studies", "aperçu", "esbozo", "thesis",
      "phd", "ma", "analysis", "sound", "tone", "tonal", "basic", "elementary", "learner", "primer", "course", "handbook",
      "reference", "xentryx", "documentation", "ethnography", "bible", "gospel", "testament"]


def text_feats_card(c):
    t = toks(c["title"])
    ts = set(t)
    f = [1.0 if k in ts else 0.0 for k in KW]
    ys = years(c["year"])
    y = ys[0] if ys else 0
    f += [len(t) / 20.0, 1.0 if not ys else 0.0, (y - 1950) / 50.0 if ys else 0.0, 1.0 if not c["creators"] else 0.0,
          c["creators"].count(" and ") / 3.0]
    return f


def entry_info(e):
    kt = set()
    for k in e["known_works"]:
        kt |= set(toks(k["title"]))
    kc = set()
    for k in e["known_works"]:
        kc |= ctoks(k["creators"])
    ky = [y for k in e["known_works"] for y in years(k["year"])]
    lab = set()
    for l in e["labels"]:
        lab |= {t for t in toks(l) if len(t) > 2 and t not in STOP}
    labtri = [tri(t) for t in lab]
    ktt = [tri(" ".join(toks(k["title"]))) for k in e["known_works"]]
    kct = [tri(c) for k in e["known_works"] for c in ctoks(k["creators"])]
    return dict(kt={t for t in kt if t not in STOP and len(t) > 2}, kc=kc, ky=ky, lab=lab, labtri=labtri,
                region=e["region"], ktt=ktt, kct=kct)


def pair_raw(ei, c, ct, cc, cy):
    ov_c = len(ei["kc"] & cc)
    ov_t = len(ei["kt"] & ct)
    jac_t = ov_t / (1 + len(ct))
    lab_ov = len(ei["lab"] & ct)
    best = 0.0
    for lt in ei["labtri"]:
        for t in ct:
            if len(t) < 3:
                continue
            tt = tri(t)
            s = len(lt & tt) / len(lt | tt)
            if s > best:
                best = s
    if cy and ei["ky"]:
        dy = min(abs(a - b) for a in cy for b in ei["ky"]) / 30.0
        hy = 0.0
    else:
        dy, hy = 0.0, 1.0
    ctt = tri(" ".join(toks(c["title"])))
    js = [len(ctt & k) / max(1, len(ctt | k)) for k in ei["ktt"]] or [0.0]
    cs = 0.0
    for a in ei["kct"]:
        for x in cc:
            b = tri(x)
            v = len(a & b) / max(1, len(a | b))
            cs = max(cs, v)
    return [float(ov_c), float(ov_c > 0), float(ov_t), jac_t, float(lab_ov), best, dy, hy, max(js), sum(js) / len(js), cs]


NPR = 25
REG = {}


def _entry_items(works):
    items = set()
    for k in works:
        items |= {"c:" + x for x in ctoks(k["creators"])}
        items |= {"t:" + t for t in toks(k["title"]) if t not in STOP and len(t) > 2 and t != "xentryx"}
    return sorted(items)[:60]


def build_region_stats(raw, extra_works=()):
    """Corpus statistics used as model inputs:
    - region association of each creator / title token / exact text,
    - within-entry co-occurrence (PMI) of creators and title tokens.
    Built from the known works of every case (public inputs) plus `extra_works`
    = (case_id, entry_id, region, work) from training answers and, in the
    self-training round, confident model predictions on unlabeled cases.
    Each case's own extra contribution is recorded so it is subtracted when
    featurizing that same case (out-of-fold), so a case never sees its own
    answers or its own predictions."""
    from collections import defaultdict, Counter
    cre, tok, txt = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    co = defaultdict(Counter)
    df = Counter()
    own = defaultdict(Counter)
    own_co = defaultdict(Counter)
    regions = Counter()
    n_ent = 0
    known = {}
    for c in raw:
        for e in c["entries"]:
            known[e["entry_id"]] = e["known_works"]
            r = e["region"]
            regions[r] += 1
            n_ent += 1
            for k in e["known_works"]:
                for x in ctoks(k["creators"]):
                    cre[x][r] += 1
                for t in set(toks(k["title"])):
                    if t not in STOP and len(t) > 2:
                        tok[t][r] += 1
                txt[(k["title"], k["creators"])][r] += 1
            items = _entry_items(e["known_works"])
            for x in items:
                df[x] += 1
                for y in items:
                    if x != y:
                        co[x][y] += 1
    by_entry = defaultdict(list)
    for cid, eid, r, k in extra_works:
        by_entry[(cid, eid, r)].append(k)
        for x in ctoks(k["creators"]):
            cre[x][r] += 1
            own[cid][("c", x, r)] += 1
        for t in set(toks(k["title"])):
            if t not in STOP and len(t) > 2:
                tok[t][r] += 1
                own[cid][("t", t, r)] += 1
        txt[(k["title"], k["creators"])][r] += 1
        own[cid][("x", (k["title"], k["creators"]), r)] += 1
    for (cid, eid, r), works in by_entry.items():
        items = _entry_items(list(known.get(eid, [])) + works)
        base = set(_entry_items(known.get(eid, [])))
        for x in items:
            for y in items:
                if x != y and not (x in base and y in base):
                    co[x][y] += 1
                    own_co[cid][(x, y)] += 1
    tot = sum(regions.values())
    REG.clear()
    REG.update(cre=cre, tok=tok, txt=txt, prior={r: v / tot for r, v in regions.items()}, own=own,
               own_co=own_co, co=co, df=df, n_ent=n_ent)


def region_feats(region, c, ct, cc, own):
    prior = REG["prior"].get(region, 0.1)

    def p(counter, kind, key):
        n = sum(counter.values())
        hit = counter.get(region, 0)
        if own:
            for r in counter:
                o = own.get((kind, key, r), 0)
                n -= o
                if r == region:
                    hit -= o
        return (hit + prior) / (n + 1), n

    best_c, nc = prior, 0
    for x in cc:
        if x in REG["cre"]:
            v, n = p(REG["cre"][x], "c", x)
            if n > nc:
                best_c, nc = v, n
    vals = []
    for t in ct:
        if t in REG["tok"]:
            v, n = p(REG["tok"][t], "t", t)
            if n >= 2:
                vals.append(math.log(v / prior))
    key = (c["title"], c["creators"])
    tv, tn = p(REG["txt"].get(key, {}), "x", key)
    return [math.log(best_c / prior), math.log1p(nc), sum(vals), max(vals) if vals else 0.0,
            math.log(tv / prior) if tn else 0.0, math.log1p(tn)]


def cooc_feats(ei, ct, cc, own_co):
    co, df, N = REG["co"], REG["df"], REG["n_ent"]
    out = []
    cards = (["c:" + x for x in cc], ["t:" + t for t in ct if t != "xentryx"])
    ents = (["c:" + x for x in ei["kc"]], ["t:" + t for t in ei["kt"] if t != "xentryx"])
    for ci in cards:
        for ej in ents:
            m, sm = 0.0, 0.0
            for x in ci:
                d = co.get(x)
                if not d:
                    continue
                for y in ej:
                    v = d.get(y, 0)
                    if v and own_co:
                        v -= own_co.get((x, y), 0)
                    if v > 0 and df[x] and df[y]:
                        pmi = math.log(v * N / (df[x] * df[y]))
                        sm += max(pmi, 0) * min(v, 3) / 3
                        m = max(m, pmi)
            out += [m, sm / (1 + len(ci))]
    return out


def build_case(c):
    own = REG["own"].get(c["case_id"])
    own_co = REG["own_co"].get(c["case_id"])
    E = c["entries"]
    C = c["cards"]
    infos = [entry_info(e) for e in E]
    cts = [{t for t in toks(x["title"]) if t not in STOP and len(t) > 2} for x in C]
    ccs = [ctoks(x["creators"]) for x in C]
    cys = [years(x["year"]) for x in C]
    raw = np.zeros((len(E), len(C), NPR), np.float32)
    for i, ei in enumerate(infos):
        for j, x in enumerate(C):
            raw[i, j] = pair_raw(ei, x, cts[j], ccs[j], cys[j]) + region_feats(ei["region"], x, cts[j], ccs[j], own) + cooc_feats(ei, cts[j], ccs[j], own_co)
    # contextual: relative to other entries for the same card and other cards for same entry
    mx_e = raw.max(0, keepdims=True)
    mx_c = raw.max(1, keepdims=True)
    others_e = np.zeros_like(raw)
    for i in range(len(E)):
        oth = np.delete(raw, i, axis=0)
        others_e[i] = oth.max(0)
    pf = np.concatenate([raw, raw - others_e, raw - mx_c, (raw == mx_e).astype(np.float32)], -1)
    cf = np.array([text_feats_card(x) for x in C], np.float32)
    # token ids
    ctok = []
    for x in C:
        ids = [h("t:" + t) for t in toks(x["title"])] + [h("c:" + t) for t in ctoks(x["creators"])]
        if TRI:
            ids += [h("g:" + g) for g in sorted(tri(" ".join(toks(x["title"]))))][:48]
        ys = years(x["year"])
        ids += [h("y:" + str(ys[0] // 10))] if ys else [h("y:none")]
        ctok.append(ids[:112] or [0])
    etok = []
    for e in E:
        ids = [h("r:" + e["region"])]
        for k in e["known_works"]:
            ids += [h("t:" + t) for t in toks(k["title"])] + [h("c:" + t) for t in ctoks(k["creators"])]
            if TRI:
                ids += [h("g:" + g) for g in sorted(tri(" ".join(toks(k["title"]))))][:48]
            ys = years(k["year"])
            if ys:
                ids.append(h("y:" + str(ys[0] // 10)))
        for l in e["labels"]:
            ids += [h("l:" + t) for t in toks(l)]
        etok.append(ids[:256] or [0])
    return dict(pf=pf, cf=cf, ctok=ctok, etok=etok, card_ids=[x["card_id"] for x in C],
                entry_ids=[e["entry_id"] for e in E])


def make_pseudo(c, rng, card_pool):
    """Self-supervised case from unlabeled inputs: hide one known work per entry
    among decoy cards from other cases; the model must find it (companion role)."""
    ents, held = [], []
    for e in c["entries"]:
        j = rng.randrange(len(e["known_works"]))
        kw = [k for i, k in enumerate(e["known_works"]) if i != j]
        ents.append(dict(entry_id=e["entry_id"] + "_p", labels=e["labels"], region=e["region"], known_works=kw))
        held.append(dict(e["known_works"][j]))
    cards = []
    for i, k in enumerate(held):
        k["card_id"] = f"p{i}"
        cards.append(k)
    while len(cards) < 18:
        x = rng.choice(card_pool)
        if x[0] != c["case_id"]:
            cards.append(dict(x[1], card_id=f"d{len(cards)}"))
    rng.shuffle(cards)
    pos = {x["card_id"]: i for i, x in enumerate(cards)}
    lab = np.array([[-1, pos[f"p{i}"], -1] for i in range(len(ents))], np.int64)
    return build_case(dict(case_id=c["case_id"] + "_p", entries=ents, cards=cards)), lab


def _ce(logits, target):
    m = target >= 0
    if m.sum() == 0:
        return logits.sum() * 0
    return F.cross_entropy(logits[m], target[m])


def pad(lists, L):
    a = np.zeros((len(lists), L), np.int64)
    for i, l in enumerate(lists):
        l = l[:L]
        a[i, :len(l)] = l
    return a


class Net(nn.Module):
    def __init__(self, npf, ncf, d=96):
        super().__init__()
        self.emb = nn.Embedding(NB, d, padding_idx=0)
        nn.init.normal_(self.emb.weight, 0, 0.02)
        self.pe = nn.Sequential(nn.Linear(d, d), nn.GELU())
        self.pc = nn.Sequential(nn.Linear(d + ncf, d), nn.GELU())
        hin = npf + ncf + 4 * d
        self.edrop = nn.Dropout(0.3)
        self.mlp = nn.Sequential(nn.Linear(hin, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 128), nn.GELU())
        self.role = nn.Linear(128, 2)
        self.typ = nn.Parameter(torch.zeros(2, d))
        self.ctx = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 2 * d, 0.1, batch_first=True,
                                                                    activation="gelu"), CTX_LAYERS)
        self.pagg = nn.Linear(npf, d)
        self.used = nn.Linear(d, 1)
        self.tier = nn.Sequential(nn.Linear(128 + d, 128), nn.GELU(), nn.Linear(128, len(TIERS)))

    def bag(self, ids):
        m = (ids > 0).float().unsqueeze(-1)
        v = (self.edrop(self.emb(ids)) * m).sum(-2) / m.sum(-2).clamp(min=1)
        return v * EMBW

    def forward(self, pf, cf, ctok, etok):
        # pf: B,5,18,P  cf: B,18,K  ctok: B,18,L  etok: B,5,L
        e = self.pe(self.bag(etok))  # B,5,d
        c = self.pc(torch.cat([self.bag(ctok), cf], -1))  # B,18,d
        B, NE, NC, _ = pf.shape
        if CTX_LAYERS > 0:
            ea = self.pagg(pf).mean(2)
            ca = self.pagg(pf).mean(1)
            z = torch.cat([e + ea + self.typ[0], c + ca + self.typ[1]], 1)
            z = self.ctx(z)
            e, c = e + z[:, :NE], c + z[:, NE:]
        self._used = self.used(c).squeeze(-1)
        ee = e.unsqueeze(2).expand(B, NE, NC, e.shape[-1])
        cc = c.unsqueeze(1).expand(B, NE, NC, c.shape[-1])
        cff = cf.unsqueeze(1).expand(B, NE, NC, cf.shape[-1])
        x = torch.cat([pf, cff, ee, cc, ee * cc, (ee - cc).abs()], -1)
        hdn = self.mlp(x)
        r = self.role(hdn)  # B,5,18,2
        return r[..., 0], r[..., 1], hdn, e

    def tier_logits(self, hdn_sel, e):
        return self.tier(torch.cat([hdn_sel, e], -1))


def to_tensors(cases, idx):
    pf = torch.tensor(np.stack([cases[i]["pf"] for i in idx]))
    cf = torch.tensor(np.stack([cases[i]["cf"] for i in idx]))
    ctok = torch.tensor(np.stack([pad(cases[i]["ctok"], 112) for i in idx]))
    etok = torch.tensor(np.stack([pad(cases[i]["etok"], 256) for i in idx]))
    return pf.to(DEV), cf.to(DEV), ctok.to(DEV), etok.to(DEV)


def train_model(cases, labels, idx, seed):
    torch.manual_seed(seed)
    random.seed(seed)
    npf = cases[idx[0]]["pf"].shape[-1]
    ncf = cases[idx[0]]["cf"].shape[-1]
    net = Net(npf, ncf).to(DEV)
    sparse_params = [net.emb.weight]
    dense = [p for n, p in net.named_parameters() if n != "emb.weight"]
    opt = torch.optim.AdamW([{"params": dense, "lr": 2e-3, "weight_decay": 1e-4},
                             {"params": sparse_params, "lr": 3e-3, "weight_decay": 0.0}])
    bs = 32
    steps = EPOCHS * math.ceil(len(idx) / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[2e-3, 3e-3], total_steps=steps, pct_start=0.1)
    for ep in range(EPOCHS):
        net.train()
        order = list(idx)
        random.shuffle(order)
        tot = 0
        for b in range(0, len(order), bs):
            bi = order[b:b + bs]
            pf, cf, ctok, etok = to_tensors(cases, bi)
            lab = torch.tensor(np.stack([labels[i] for i in bi])).to(DEV)  # B,5,3
            ls, cs, hdn, e = net(pf, cf, ctok, etok)
            B, NE, NC = ls.shape
            real = lab[:, 0, 0] >= 0
            l1 = _ce(ls.reshape(B * NE, NC), lab[..., 0].reshape(-1))
            l2 = _ce(cs.reshape(B * NE, NC), lab[..., 1].reshape(-1))
            l4 = F.cross_entropy(cs.transpose(1, 2).reshape(B * NC, NE)[_cardmask(lab[..., 1], NC)],
                                 _cardtarget(lab[..., 1], NC))
            loss = l1 + l2 + 0.3 * l4
            if real.any():
                lr_, ls_, hd_, e_, u_ = lab[real], ls[real], hdn[real], e[real], net._used[real]
                Br = lr_.shape[0]
                l3 = F.cross_entropy(ls_.transpose(1, 2).reshape(Br * NC, NE)[_cardmask(lr_[..., 0], NC)],
                                     _cardtarget(lr_[..., 0], NC))
                hsel = hd_.gather(2, lr_[..., 0].view(Br, NE, 1, 1).expand(Br, NE, 1, hd_.shape[-1])).squeeze(2)
                tl = net.tier_logits(hsel, e_)
                l5 = F.cross_entropy(tl.reshape(Br * NE, -1), lr_[..., 2].reshape(-1))
                ut = torch.zeros(Br, NC, device=DEV)
                ut.scatter_(1, lr_[..., 0], 1.0)
                ut.scatter_(1, lr_[..., 1], 1.0)
                l6 = F.binary_cross_entropy_with_logits(u_, ut)
                loss = loss + 0.3 * l3 + l5 + 0.5 * l6
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(bi)
        print(f"  seed {seed} ep {ep} loss {tot / len(idx):.4f}", flush=True)
    return net


def _cardmask(lab, NC):
    B, NE = lab.shape
    m = torch.zeros(B, NC, dtype=torch.bool, device=lab.device)
    m.scatter_(1, lab, True)
    return m.reshape(-1)


def _cardtarget(lab, NC):
    B, NE = lab.shape
    t = torch.full((B, NC), -1, dtype=torch.long, device=lab.device)
    t.scatter_(1, lab, torch.arange(NE, device=lab.device).expand(B, NE).contiguous())
    t = t.reshape(-1)
    return t[t >= 0]


@torch.no_grad()
def predict(nets, cases, idx):
    out = {}
    for b in range(0, len(idx), 64):
        bi = idx[b:b + 64]
        pf, cf, ctok, etok = to_tensors(cases, bi)
        LP = 0
        CP = 0
        rets = []
        for net in nets:
            net.eval()
            ls, cs, hdn, e = net(pf, cf, ctok, etok)
            LP = LP + F.log_softmax(ls, -1) / len(nets)
            CP = CP + F.log_softmax(cs, -1) / len(nets)
            rets.append((net, hdn, e))
        LP = LP.cpu().numpy()
        CP = CP.cpu().numpy()
        for k, ci in enumerate(bi):
            NE, NC = LP[k].shape
            cost = np.concatenate([-LP[k], -CP[k]], 0)  # 2NE x NC
            r, cidx = linear_sum_assignment(cost)
            lead = np.zeros(NE, np.int64)
            comp = np.zeros(NE, np.int64)
            for rr, cc in zip(r, cidx):
                if rr < NE:
                    lead[rr] = cc
                else:
                    comp[rr - NE] = cc
            tp = 0
            li = torch.tensor(lead, device=DEV)
            for net, hdn, e in rets:
                hsel = hdn[k][torch.arange(NE, device=DEV), li]
                tp = tp + F.softmax(net.tier_logits(hsel, e[k]), -1)
            tiers = tp.argmax(-1).cpu().numpy()
            cs_ = cases[ci]
            for i in range(NE):
                out[cs_["entry_ids"][i]] = dict(lead_id=cs_["card_ids"][lead[i]], companion_id=cs_["card_ids"][comp[i]],
                                                tier=TIERS[tiers[i]], lead_p=float(np.exp(LP[k][i, lead[i]])),
                                                comp_p=float(np.exp(CP[k][i, comp[i]])))
    return out


def score(pred, gold, entry_case):
    from collections import defaultdict
    byc = defaultdict(list)
    for eid, g in gold.items():
        p = pred[eid]
        L = p["lead_id"] == g["lead_id"]
        R = p["companion_id"] == g["companion_id"]
        T = p["tier"] == g["tier"]
        byc[entry_case[eid]].append((L, R, T))
    tot = 0
    for c, rows in byc.items():
        Es = [0.1 * L + 0.1 * R + 0.1 * T + 0.7 * (L and R and T) for L, R, T in rows]
        J = [L and R and T for L, R, T in rows]
        n = len(J)
        pairs = [(J[a] and J[b]) for a in range(n) for b in range(a + 1, n)]
        P = sum(pairs) / len(pairs) if pairs else float(J[0])
        tot += 0.8 * np.mean(Es) + 0.2 * P
    return 100 * tot / len(byc)


def main():
    public_dir = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    print("device", DEV, flush=True)
    train = pd.read_csv(public_dir / "train.csv")
    test = pd.read_csv(public_dir / "test.csv")
    tt = pd.read_csv(public_dir / "train_targets.csv")
    gold = {r.target_id: json.loads(r.prediction) for r in tt.itertuples()}
    raw = [json.loads(l) for l in open(public_dir / "cases.jsonl", encoding="utf8")]
    raw_by_id = {c["case_id"]: c for c in raw}
    train_cases = sorted(set(train.case_id))
    test_cases = sorted(set(test.case_id))
    all_ids = train_cases + test_cases
    if VAL:
        tc = list(train_cases)
        random.Random(0).shuffle(tc)
        target, fit = tc[:200], tc[200:]
    else:
        target, fit = test_cases, train_cases
    egold, entry_case = {}, {}
    for r in train.itertuples():
        egold[r.entry_id] = gold[r.target_id]
        entry_case[r.entry_id] = r.case_id
    ereg = {e["entry_id"]: e["region"] for c in raw for e in c["entries"]}
    cmap = {c["case_id"]: {x["card_id"]: x for x in c["cards"]} for c in raw}
    fit_set = set(fit)
    gold_works = []
    if USE_GOLD_STATS:
        for r in train.itertuples():
            if r.case_id in fit_set:
                g = gold[r.target_id]
                for role in ("lead_id", "companion_id"):
                    gold_works.append((r.case_id, r.entry_id, ereg[r.entry_id], cmap[r.case_id][g[role]]))

    def featurize(extra):
        build_region_stats(raw, extra)
        cs = {cid: build_case(raw_by_id[cid]) for cid in all_ids}
        print("built", len(cs), "extra works", len(extra), flush=True)
        return cs

    labels = {}
    cases = featurize(gold_works)
    for cid in fit:
        cs = cases[cid]
        pos = {x: j for j, x in enumerate(cs["card_ids"])}
        labels[cid] = np.array([[pos[egold[e]["lead_id"]], pos[egold[e]["companion_id"]], TIERS.index(egold[e]["tier"])]
                                for e in cs["entry_ids"]], np.int64)

    def run_round(cases, tag):
        nets = [train_model(cases, labels, list(fit), SEED + s) for s in range(N_MODELS)]
        pred = predict(nets, cases, list(target))
        if VAL:
            vg = {e: egold[e] for c in target for e in cases[c]["entry_ids"]}
            L = np.mean([pred[e]["lead_id"] == vg[e]["lead_id"] for e in vg])
            R = np.mean([pred[e]["companion_id"] == vg[e]["companion_id"] for e in vg])
            T = np.mean([pred[e]["tier"] == vg[e]["tier"] for e in vg])
            print(tag, "VAL score", score(pred, vg, entry_case), "lead", L, "comp", R, "tier", T, flush=True)
        return pred

    pred = run_round(cases, "round1")
    for rnd in range(SELF_TRAIN):
        # self-training on unlabeled target cases: confident predictions become
        # extra corpus works (never used for the case that produced them)
        extra = list(gold_works)
        for cid in target:
            for eid in cases[cid]["entry_ids"]:
                p = pred[eid]
                if p["lead_p"] >= CONF:
                    extra.append((cid, eid, ereg[eid], cmap[cid][p["lead_id"]]))
                if p["comp_p"] >= CONF:
                    extra.append((cid, eid, ereg[eid], cmap[cid][p["companion_id"]]))
        cases = featurize(extra)
        pred = run_round(cases, f"round{rnd + 2}")
    if VAL:
        return
    rows = []
    for r in test.itertuples():
        p = pred[r.entry_id]
        rows.append((r.target_id, json.dumps({"lead_id": p["lead_id"], "companion_id": p["companion_id"],
                                              "tier": p["tier"]})))
    sub = pd.DataFrame(rows, columns=["target_id", "prediction"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False)
    print("wrote", out_path, len(sub))


if __name__ == "__main__":
    main()
