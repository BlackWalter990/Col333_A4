import re
import sys
from pprint import pprint

def parse_bif(filename):
    with open(filename, 'r') as f:
        text = f.read()
    
    var_pattern = re.compile(
        r'variable\s+(\w+)\s*\{\s*type\s+discrete\s*\[\s*\d+\s*\]\s*=\s*\{\s*([^\}]+)\s*\}\s*;',
        re.MULTILINE
    )

    variables = {}
    for name, values_str in var_pattern.findall(text):
        values = [v.strip() for v in re.split(r'[,\s]+', values_str) if v.strip()]
        variables[name] = {
            "values": values,
            "parents": [],
            "probabilities": {}
        }
    prob_pattern = re.compile(
        r'probability\s*\(\s*([^\|\)]+?)\s*(?:\|\s*([^\)]+))?\s*\)\s*\{(.*?)\};',
        re.DOTALL | re.MULTILINE
    )

    for target, parents_str, body in prob_pattern.findall(text):
        target = target.strip()
        parents = [p.strip() for p in parents_str.split(',')] if parents_str else []

        if target not in variables:
            variables[target] = {"values": [], "parents": parents, "probabilities": {}}
        else:
            variables[target]["parents"] = parents

        # Clean body
        body = body.strip()
        if body.startswith("table"):
            # Simple unconditional case
            table_vals = re.findall(r'[-\d\.]+', body)
            probs = [float(v) if v != '-1' else -1 for v in table_vals]
            variables[target]["probabilities"]["()"] = probs
        else:
            # Conditional block with multiple parent value rows
            row_pattern = re.compile(r'\(([^\)]+)\)\s*([^\n;]+)')
            for parent_vals, probs_str in row_pattern.findall(body):
                key = tuple(v.strip() for v in parent_vals.split(','))
                probs = [float(x) if x != '-1' else -1 for x in re.findall(r'[-\d\.]+', probs_str)]
                variables[target]["probabilities"][key] = probs

    return variables



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python starter.py <hailfinder.bif>")
        sys.exit(1)

    bif_data = parse_bif(sys.argv[1])
    print(f"Parsed {len(bif_data)} variables.\n")

    # Summary


