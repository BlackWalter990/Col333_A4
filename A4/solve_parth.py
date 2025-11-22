import sys
from startup_code import parse_bif


def parse_csv_data(filename):
    parsed = []
    with open(filename, 'r') as f:
        for line in f:
            fields = [token[1:-1] for token in line.strip().split(',')]
            parsed.append(fields)
    return parsed


def learn_parameters(network, data):
    var_index = {var: idx for idx, var in enumerate(network.keys())}

    children = {var: [] for var in network.keys()}
    for child, desc in network.items():
        for parent in desc['parents']:
            children[parent].append(child)

    # Helpers
    def get_root_cfg(var):
        return next(iter(network[var]['probabilities'].keys()))

    def normalize_row(row):
        if any(x < 0 for x in row):
            return [1.0 / len(row)] * len(row)
        total = sum(row)
        if total <= 0:
            return [1.0 / len(row)] * len(row)
        return [x / total for x in row]

    def get_cpt_row(var, cfg):
        """Return normalized CPT row for variable given parent config."""
        probs = network[var]['probabilities']
        if network[var]['parents']:
            row = probs[cfg]
        else:
            row = next(iter(probs.values()))
        return normalize_row(row)

    # Initialize counts
    counts = {
        var: {cfg: [0.0] * len(desc['values'])
              for cfg in desc['probabilities']}
        for var, desc in network.items()
    }

    # Process each record
    for record in data:
        # Detect missing variable
        missing_idx = next((i for i, v in enumerate(record) if v == "?"), None)

        # Case 1: No missing value
        if missing_idx is None:
            for var, desc in network.items():
                parents = desc['parents']
                values = desc['values']

                cfg = tuple(record[var_index[p]] for p in parents) if parents else get_root_cfg(var)
                value = record[var_index[var]]

                counts[var][cfg][values.index(value)] += 1.0
            continue

        # Case 2: One missing value
        missing_var = list(network.keys())[missing_idx]
        descX = network[missing_var]
        valuesX = descX['values']
        parentsX = descX['parents']

        parent_cfg_X = tuple(record[var_index[p]] for p in parentsX) if parentsX else get_root_cfg(missing_var)
        prior = get_cpt_row(missing_var, parent_cfg_X)

        # Compute posterior weights
        weights = [0.0] * len(valuesX)
        direct_children = children[missing_var]

        for val_idx, val in enumerate(valuesX):
            weight = prior[val_idx]
            if weight == 0:
                continue

            # Multiply likelihood from children
            for child in direct_children:
                child_desc = network[child]
                child_cfg = tuple(
                    val if p == missing_var else record[var_index[p]]
                    for p in child_desc['parents']
                )
                row = get_cpt_row(child, child_cfg)

                observed_child_val = record[var_index[child]]
                child_pos = child_desc['values'].index(observed_child_val)

                weight *= row[child_pos]
                if weight == 0:
                    break

            weights[val_idx] = weight
        Z = sum(weights)
        if Z <= 0:
            gamma = [1.0 / len(valuesX)] * len(valuesX)
        else:
            gamma = [w / Z for w in weights]

        for val_idx, g in enumerate(gamma):
            counts[missing_var][parent_cfg_X][val_idx] += g

        for val_idx, g in enumerate(gamma):
            val = valuesX[val_idx]
            for child in direct_children:
                child_desc = network[child]
                child_cfg = tuple(
                    val if p == missing_var else record[var_index[p]]
                    for p in child_desc['parents']
                )
                observed = record[var_index[child]]
                pos = child_desc['values'].index(observed)
                counts[child][child_cfg][pos] += g

        affected = set(direct_children) | {missing_var}
        for var, desc in network.items():
            if var in affected:
                continue
            cfg = tuple(record[var_index[p]] for p in desc['parents']) if desc['parents'] else get_root_cfg(var)
            val = record[var_index[var]]
            pos = desc['values'].index(val)
            counts[var][cfg][pos] += 1.0

    for var, desc in network.items():
        for cfg, row_counts in counts[var].items():
            total = sum(row_counts)
            if total > 0:
                network[var]['probabilities'][cfg] = [c / total for c in row_counts]
            else:
                uniform = 1.0 / len(desc['values'])
                network[var]['probabilities'][cfg] = [uniform] * len(desc['values'])

    return network


def write_bif(network, filename):
    """Write Bayesian network to a .bif file."""
    def fmt_num(x):
        s = f"{float(x):.6f}".rstrip('0').rstrip('.')
        return s or "0"

    def fmt_row(row):
        return ", ".join(fmt_num(p) for p in row)

    with open(filename, "w") as f:
        f.write('network "Unknown" {\n}\n')

        for var, info in network.items():
            values = ", ".join(info["values"])
            f.write(f"variable {var} {{\n")
            f.write(f"  type discrete [ {len(info['values'])} ] = {{ {values} }};\n")
            f.write("}\n")

        for var, info in network.items():
            parents = info['parents']
            probs = info['probabilities']

            if parents:
                parent_list = ", ".join(parents)
                f.write(f"probability ( {var} | {parent_list} ) {{\n")
                for cfg, row in probs.items():
                    inst = ", ".join(cfg) if isinstance(cfg, (list, tuple)) else cfg
                    f.write(f"  ( {inst} ) {fmt_row(row)};\n")
                f.write("};\n")
            else:
                row = next(iter(probs.values()))
                f.write(f"probability ( {var} ) {{\n")
                f.write(f"  table {fmt_row(row)};\n")
                f.write("};\n")


def main():
    if len(sys.argv) != 3:
        print("Incorrect input format")
        return

    bif_file, data_file = sys.argv[1], sys.argv[2]
    network = parse_bif(bif_file)
    data = parse_csv_data(data_file)

    learned_network = learn_parameters(network, data)
    write_bif(learned_network, 'solved_hailfinder.bif')


if __name__ == "__main__":
    main()