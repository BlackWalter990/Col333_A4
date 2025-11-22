#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust EM learner for Bayesian Networks (<=1 missing per record).
Parses both strict and loose BIF variants used in COL333 kits.

Usage:
  python solve.py hailfinder.bif records.dat
Writes:
  solved_hailfinder.bif
"""

import sys
import re
from itertools import product
from collections import defaultdict

# ---------------------------
# Data structures
# ---------------------------

class Variable:
    def __init__(self, name, values):
        self.name = name
        self.values = values

class CPT:
    def __init__(self, child, parents, child_dom, parent_domains, flat):
        self.child = child
        self.parents = parents[:]                 # ordered
        self.child_dom = child_dom[:]             # value order
        self.parent_domains = [parent_domains[p] for p in parents]
        self.parent_cards = [len(dom) for dom in self.parent_domains]
        self.child_card = len(child_dom)

        self.flat = flat[:]                       # list of floats (may include -1)
        self.known_mask = [False if v < 0 else True for v in self.flat]

        # fast parent value -> index
        self.pval_index = [{v:i for i,v in enumerate(dom)} for dom in self.parent_domains]

    def parent_config_count(self):
        n = 1
        for c in self.parent_cards:
            n *= c
        return n

    def parent_to_offset(self, parent_tuple):
        # rightmost parent varies fastest
        idx = 0
        stride = 1
        for j in range(len(self.parents)-1, -1, -1):
            v = parent_tuple[j]
            vidx = self.pval_index[j][v]
            idx += vidx * stride
            stride *= self.parent_cards[j]
        return idx * self.child_card

    def iter_parent_tuples(self):
        if not self.parents:
            yield tuple(), 0
            return
        for combo in product(*self.parent_domains):
            yield combo, self.parent_to_offset(combo)

class BN:
    def __init__(self):
        self.variables = []                  # [Variable]
        self.var_index = {}                  # name -> idx
        self.domains = {}                    # name -> [values]
        self.cpts = {}                       # name -> CPT
        self.parents = defaultdict(list)     # name -> [parents]
        self.children = defaultdict(list)    # name -> [children]

    def add_var(self, name, values):
        self.var_index[name] = len(self.variables)
        self.variables.append(Variable(name, values))
        self.domains[name] = values

    def add_cpt(self, child, parents, flat):
        cpt = CPT(child, parents, self.domains[child], self.domains, flat)
        self.cpts[child] = cpt
        self.parents[child] = parents[:]
        for p in parents:
            self.children[p].append(child)

    def order(self):
        return [v.name for v in self.variables]

# ---------------------------
# BIF parsing (tolerant)
# ---------------------------

def _strip_bom_comments(txt):
    txt = txt.lstrip("\ufeff")
    # kill //... and /* ... */
    txt = re.sub(r"//.*?$", "", txt, flags=re.MULTILINE)
    txt = re.sub(r"/\*.*?\*/", "", txt, flags=re.DOTALL)
    return txt

def parse_bif(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()
    txt = _strip_bom_comments(txt)

    bn = BN()

    # Variables: accept both styles.
    # Style A (loose kit): variable X { type discrete[n] = { a, b, c } ; }
    var_pat_loose = re.compile(
        r'variable\s+([A-Za-z_]\w*)\s*\{[^}]*?discrete\s*\[\s*(\d+)\s*\]\s*=\s*\{([^}]*)\}[^}]*?\}',
        re.IGNORECASE | re.DOTALL
    )
    # Style B (strict Hugin): variable "X" { type discrete[n] { "a" "b" } ; }
    var_pat_strict = re.compile(
        r'variable\s+"([^"]+)"\s*\{[^}]*?discrete\s*\[\s*(\d+)\s*\]\s*\{([^}]*)\}[^}]*?\}',
        re.IGNORECASE | re.DOTALL
    )

    found_vars = False

    for m in var_pat_loose.finditer(txt):
        name = m.group(1)
        card = int(m.group(2))
        vals_blob = m.group(3)
        # values can be unquoted, comma/space separated
        vals = [v.strip().strip(',') for v in re.split(r'[,\s]+', vals_blob) if v.strip() and v.strip() != ',']
        if len(vals) != card:
            raise ValueError(f"Cardinality mismatch for variable {name}: declared {card}, got {len(vals)}")
        bn.add_var(name, vals)
        found_vars = True

    if not found_vars:
        for m in var_pat_strict.finditer(txt):
            name = m.group(1)
            card = int(m.group(2))
            vals_blob = m.group(3)
            vals = re.findall(r'"([^"]+)"', vals_blob)
            vals = [v.strip().strip(',') for v in vals]
            if len(vals) != card:
                raise ValueError(f"Cardinality mismatch for variable {name}: declared {card}, got {len(vals)}")
            bn.add_var(name, vals)
            found_vars = True

    # Probabilities: accept both header forms and any table spacing.
    # Style with pipe/commas: probability ( X | P1, P2 ) { table ... ; }
    prob_pat_pipe = re.compile(
        r'probability\s*\(\s*([A-Za-z_]\w*|\"[^"]+\")\s*(?:\|\s*([^)]*))?\)\s*\{[^}]*?table\s+([^;]+);',
        re.IGNORECASE | re.DOTALL
    )
    # Style list of quoted names: probability ( "X" "P1" "P2" ) { table ... ; }
    prob_pat_list = re.compile(
        r'probability\s*\(\s*(\"[^"]+\"(?:\s+\"[^"]+\")*)\s*\)\s*\{[^}]*?table\s+([^;]+);',
        re.IGNORECASE | re.DOTALL
    )

    def norm_name(tok):
        tok = tok.strip()
        return tok[1:-1] if tok.startswith('"') and tok.endswith('"') else tok

    # First try pipe form
    matched = False
    for m in prob_pat_pipe.finditer(txt):
        child_tok = m.group(1)
        parents_blob = m.group(2) or ""
        nums_blob = m.group(3)

        child = norm_name(child_tok)
        parents = [norm_name(t) for t in re.split(r'\s*,\s*', parents_blob.strip()) if t.strip()] if parents_blob.strip() else []

        nums = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', nums_blob)
        flat = [float(x) for x in nums]
        if child not in bn.domains:
            # tolerate forward probability blocks by skipping until variable seen
            continue
        bn.add_cpt(child, parents, flat)
        matched = True

    # Then the list-of-quoted-names form
    for m in prob_pat_list.finditer(txt):
        names_blob = m.group(1)
        nums_blob = m.group(2)
        names = [norm_name(s) for s in re.findall(r'"([^"]+)"', names_blob)]
        if not names:
            continue
        child = names[0]
        parents = names[1:]
        nums = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', nums_blob)
        flat = [float(x) for x in nums]
        if child not in bn.domains:
            continue
        # avoid duplicate if already added via pipe pattern
        if child in bn.cpts and bn.cpts[child].parents == parents and len(bn.cpts[child].flat) == len(flat):
            continue
        bn.add_cpt(child, parents, flat)
        matched = True

    return bn

# ---------------------------
# Records
# ---------------------------

def load_records(path, var_order, domains):
    recs = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            toks = [t for t in re.split(r'\s+', line) if t]
            cleaned = []
            for tok in toks:
                tok = tok.strip().strip(',').strip()
                if tok.startswith('"') and tok.endswith('"'):
                    tok = tok[1:-1]
                tok = tok.replace(',', '')
                cleaned.append(tok)
            if len(cleaned) != len(var_order):
                raise ValueError(f"Record length {len(cleaned)} != {len(var_order)}")
            # domain sanity
            for i, v in enumerate(cleaned):
                if v != '?' and v not in domains[var_order[i]]:
                    raise ValueError(f"Unknown value '{v}' for {var_order[i]}")
            recs.append(cleaned)
    return recs

# ---------------------------
# EM
# ---------------------------

class EMLearner:
    def __init__(self, bn):
        self.bn = bn
        self.params = {}      # node -> {pt: [probs]}
        self.known = {}       # node -> {pt: [bool]}
        self.var_order = bn.order()
        self.vi = {n:i for i,n in enumerate(self.var_order)}
        self.val_index = {n:{v:i for i,v in enumerate(bn.domains[n])} for n in self.var_order}
        self.parent_tuple_list = {}  # node -> [pt tuples]
        self._init_params()

    def _init_params(self):
        for child, cpt in self.bn.cpts.items():
            tuples = [pt for pt,_ in cpt.iter_parent_tuples()]
            self.parent_tuple_list[child] = tuples
            params = {}
            known = {}
            k = len(self.bn.domains[child])
            for pt,_ in cpt.iter_parent_tuples():
                off = cpt.parent_to_offset(pt)
                row = cpt.flat[off: off+k]
                mask = cpt.known_mask[off: off+k]
                if not all(mask):
                    known_mass = sum(p for p,m in zip(row,mask) if m)
                    unk = [i for i,m in enumerate(mask) if not m]
                    rem = max(0.0, 1.0 - known_mass)
                    fill = rem/len(unk) if unk else 0.0
                    for i in unk:
                        row[i] = fill
                s = sum(max(0.0,p) for p in row)
                row = [1.0/k]*k if s <= 0 else [max(0.0,p)/s for p in row]
                params[pt] = row
                known[pt] = mask
            self.params[child] = params
            self.known[child] = known

    def _miss_index(self, row):
        m = -1
        for i,v in enumerate(row):
            if v == '?':
                if m != -1:
                    raise ValueError("More than one missing value in a row, violates spec.")
                m = i
        return m

    def _pt(self, node, row):
        return tuple(row[self.vi[p]] for p in self.bn.parents[node])

    def _posterior(self, miss_idx, row):
        X = self.var_order[miss_idx]
        dom = self.bn.domains[X]
        Xc = self.bn.cpts[X]
        pa_vals = [row[self.vi[p]] for p in Xc.parents]

        # unnormalized weights: P(X|Pa) * prod_{Y in children(X)} P(y_obs | Pa(Y))
        w = []
        for xv in dom:
            xi = self.val_index[X][xv]
            pX = self.params[X][tuple(pa_vals)][xi]
            val = pX
            for Y in self.bn.children[X]:
                Yc = self.bn.cpts[Y]
                y_obs = row[self.vi[Y]]
                y_idx = self.val_index[Y][y_obs]
                y_pa = []
                for p in Yc.parents:
                    y_pa.append(xv if p == X else row[self.vi[p]])
                val *= self.params[Y][tuple(y_pa)][y_idx]
            w.append(val)
        s = sum(w)
        return [1.0/len(dom)]*len(dom) if s <= 0 else [x/s for x in w]

    def fit(self, records, max_iter=50, tol=1e-6):
        for _ in range(max_iter):
            counts = {node:{pt:[0.0]*len(self.bn.domains[node]) for pt in self.parent_tuple_list[node]}
                      for node in self.params}
            # E-step
            for row in records:
                mi = self._miss_index(row)
                if mi == -1:
                    # hard counts
                    for node in self.var_order:
                        vi = self.val_index[node][row[self.vi[node]]]
                        counts[node][self._pt(node,row)][vi] += 1.0
                else:
                    post = self._posterior(mi, row)
                    X = self.var_order[mi]
                    dom = self.bn.domains[X]
                    for node in self.var_order:
                        if node == X:
                            pt = self._pt(node, row)
                            for i, w in enumerate(post):
                                counts[node][pt][i] += w
                        else:
                            v_obs = row[self.vi[node]]
                            vi_obs = self.val_index[node][v_obs]
                            cpt = self.bn.cpts[node]
                            if X in cpt.parents:
                                par = cpt.parents
                                base = [row[self.vi[p]] for p in par]
                                for i,xv in enumerate(dom):
                                    tmp = list(base)
                                    for j,p in enumerate(par):
                                        if p == X:
                                            tmp[j] = xv
                                            break
                                    counts[node][tuple(tmp)][vi_obs] += post[i]
                            else:
                                counts[node][self._pt(node,row)][vi_obs] += 1.0
            # M-step
            max_delta = 0.0
            for node, pts in self.params.items():
                known = self.known[node]
                for pt, old in pts.items():
                    krow = known[pt]
                    new = [0.0]*len(old)
                    known_sum = 0.0
                    for i,is_known in enumerate(krow):
                        if is_known:
                            new[i] = old[i]
                            known_sum += new[i]
                    remain = max(0.0, 1.0 - known_sum)
                    unk = [i for i,k in enumerate(krow) if not k]
                    if unk:
                        s = sum(max(0.0, counts[node][pt][i]) for i in unk)
                        if s <= 0:
                            fill = remain/len(unk)
                            for i in unk:
                                new[i] = fill
                        else:
                            for i in unk:
                                new[i] = remain * max(0.0, counts[node][pt][i]) / s
                    else:
                        s = sum(new)
                        if s > 0:
                            new = [p/s for p in new]
                        else:
                            new = old[:]
                    for a,b in zip(old,new):
                        if abs(a-b) > max_delta:
                            max_delta = abs(a-b)
                    self.params[node][pt] = new
            if max_delta < tol:
                break

    def write_bif(self, out_path):
        # Emit a consistent, permissive style:
        # variable X { type discrete[K] = { v1, v2, ... } ; }
        # probability ( X | P1, P2 ) { table ... ; }
        out = []
        out.append("// solved_hailfinder.bif\n")
        for v in self.bn.variables:
            vals = ", ".join(v.values)
            out.append(f"variable {v.name} {{")
            out.append(f"  type discrete[{len(v.values)}] = {{ {vals} }} ;")
            out.append("}\n")
        for child in self.bn.order():
            parents = self.bn.parents[child]
            header = f"probability ( {child} ) {{" if not parents else f"probability ( {child} | {', '.join(parents)} ) {{"
            out.append(header)
            cpt = self.bn.cpts[child]
            dom_k = len(self.bn.domains[child])
            row_vals = []
            for pt,_ in cpt.iter_parent_tuples():
                probs = self.params[child][pt]
                # round to 4 decimals, fix drift by adjusting argmax
                rounded = [round(p + 1e-12, 4) for p in probs]
                diff = round(1.0 - sum(rounded), 4)
                if abs(diff) >= 1e-4:
                    k = max(range(len(rounded)), key=lambda i: rounded[i])
                    rounded[k] = round(rounded[k] + diff, 4)
                row_vals.extend(f"{x:.4f}" for x in rounded)
            out.append("  table " + " ".join(row_vals) + " ;")
            out.append("}\n")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(out))

# ---------------------------
# Main
# ---------------------------

def main():
    if len(sys.argv) != 3:
        print("Usage: python solve.py hailfinder.bif records.dat")
        sys.exit(1)
    bif_path = sys.argv[1]
    dat_path = sys.argv[2]

    bn = parse_bif(bif_path)
    if len(bn.variables) == 0:
        print("ERROR: no variables parsed from BIF.")
        sys.exit(1)
    if any(name not in bn.cpts for name in bn.order()):
        missing = [n for n in bn.order() if n not in bn.cpts]
        print("ERROR: missing probability blocks for:", ", ".join(missing))
        sys.exit(1)

    records = load_records(dat_path, bn.order(), bn.domains)
    if len(records) == 0:
        print("ERROR: no records loaded.")
        sys.exit(1)

    learner = EMLearner(bn)
    learner.fit(records)
    learner.write_bif("solved_hailfinder.bif")

if __name__ == "__main__":
    main()