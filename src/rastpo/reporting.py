"""Build the paper's tables from saved experiment results."""

import csv
import json
from pathlib import Path

import numpy as np

from .data import save_json


RULES = {"oscar": "OSCAR", "sdp": "SD-relaxation", "pga": "mSSRM-PGA", "afba": "ASMP-AFBA"}
LEARNED = [(f"pfl_{key}", f"PFL / {value}") for key, value in RULES.items()] + [
    ("dfstpo", "DF-STPO"), ("rastpo", "RA-STPO (Ours)")]
METHODS = [(f"historic_{key}", f"Historic / {value}") for key, value in RULES.items()] + LEARNED


def read(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def tex(value):
    return str(value).replace("&", r"\&").replace("_", r"\_").replace("%", r"\%")


def number(value, digits=3):
    return "--" if value is None else f"{value:.{digits}f}"


def markets_in(record):
    return {} if record is None else {row["name"]: row for row in record["markets"]}


class Tables:
    def __init__(self, root, markets):
        self.root = root / "tables"
        self.root.mkdir(exist_ok=True)
        self.markets = markets
        self.index = []

    def write(self, name, headers, rows, body, columns, *, groups=None, description=""):
        with (self.root / (name + ".csv")).open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(headers)
            writer.writerows(rows)
        lines = [r"\begin{tabular}{" + columns + "}", r"\toprule"]
        if groups:
            lines.extend(groups)
        lines.extend(body)
        lines += [r"\bottomrule", r"\end{tabular}"]
        (self.root / (name + ".tex")).write_text("\n".join(lines) + "\n")
        self.index.append(dict(name=name, rows=len(rows), description=description))

    def performance(self, name, entries, *, rank=False, description=""):
        """Entries are (label, market -> metric dict); each market has SR and W."""
        if not any(values for _, values in entries):
            return
        rows, body = [], []
        leaders = {}
        if rank:
            for market in self.markets:
                for metric in ["sharpe", "wealth"]:
                    values = {v[market][metric] for _, v in entries
                              if market in v and v[market].get(metric) is not None}
                    leaders[(market, metric)] = sorted(values, reverse=True)[:2]
        sensitivity = name == "tab_mj_performance"
        width = 2 if sensitivity else 1
        groups = [(" & " if sensitivity else "Method") + " & " + " & ".join(
            r"\multicolumn{2}{c}{" + tex(m) + "}" for m in self.markets) + r" \\",
            "".join(r"\cmidrule(lr){" + f"{width+1+2*i}-{width+2+2*i}" + "}"
                    for i in range(len(self.markets))),
            (r"$J$ & $M$" if sensitivity else "") + " & " +
            " & ".join([r"SR & $W$"] * len(self.markets)) + r" \\", r"\midrule"]
        previous_group = None
        for label, values in entries:
            group, short = None, label
            if name == "tab_main":
                if label.startswith("Historic / "):
                    group, short = "Historic", label.split(" / ",1)[1]
                elif label.startswith("PFL / "):
                    group, short = "Prediction-focused learning", label.split(" / ",1)[1]
                else:
                    group = "Decision-focused learning"
            elif name == "tab_input_comparison":
                short, rep = label.rsplit(" / ",1)
                group = "Window-derived inputs" if rep == "window" else "Raw return inputs"
            elif name == "tab_sign":
                short, group = label.rsplit(" / ",1)
            if group is not None and group != previous_group:
                body.append(r"\multicolumn{" + str(1+2*len(self.markets)) + r"}{l}{\emph{" + tex(group) + r"}} \\")
                previous_group = group
            display = r"\textbf{" + tex(short) + "}" if short == "RA-STPO (Ours)" else tex(short)
            if group is not None:
                display = r"\hspace{7pt}" + display
            row_key = [int(part.split("=")[1]) for part in label.split(", ")] if sensitivity else [label]
            cells = [str(v) for v in row_key] if sensitivity else [display]
            for market in self.markets:
                metrics = values.get(market, {})
                rows.append([*row_key, market, metrics.get("sharpe"), metrics.get("wealth")])
                for metric in ["sharpe", "wealth"]:
                    value = metrics.get(metric)
                    cell = number(value)
                    best = leaders.get((market, metric), [])
                    if value is not None and value in best:
                        cell = r"\textbf{" + cell + "}"
                        if value == best[0]:
                            cell = r"\underline{" + cell + "}"
                    cells.append(cell)
            body.append(" & ".join(cells) + r" \\")
        headers = (["J", "M"] if sensitivity else ["method"]) + ["market", "SR", "W"]
        self.write(name, headers, rows, body,
                   "@{}" + ("cc" if sensitivity else "l") + "cc" * len(self.markets) + "@{}",
                   groups=groups, description=description)


def summarize(output, data, *, members=16, baseline_members=32, faces=8,
              bootstrap_replicates=100, sensitivity_members=(1,4,8,16,32),
              face_counts=(1,2,4,8,16), benchmark_markets=None):
    root = Path(output)
    manifest = read(Path(data) / "dataset.json")
    if manifest is None:
        raise ValueError("summarize needs a prepared window dataset (--data)")
    cfg = dict(data=str(Path(data).resolve()), members=members,
               baseline_members=baseline_members, faces=faces,
               bootstrap_replicates=bootstrap_replicates,
               sensitivity_members=list(sensitivity_members), face_counts=list(face_counts),
               benchmark_markets=benchmark_markets)
    root.mkdir(parents=True, exist_ok=True)
    names = [row["name"] for row in manifest["markets"]]
    tables = Tables(root, names)

    def evaluation(path):
        return markets_in(read(root / "evaluations" / path / "evaluation.json"))

    def values(path):
        return {key: row["mean"] for key, row in evaluation(path).items()}

    def simple(name, headers, rows, formats=None, description=""):
        if not rows:
            return
        formats = formats or [None] * len(headers)
        body = [" & ".join(tex(h) for h in headers) + r" \\", r"\midrule"]
        for row in rows:
            cells = [tex(v) if fmt is None else ("--" if v is None else format(v, fmt))
                     for v, fmt in zip(row, formats)]
            body.append(" & ".join(cells) + r" \\")
        tables.write(name, headers, rows, body, "l" + "r" * (len(headers)-1), description=description)

    tables.performance("tab_main", [(label, values(f"window/main/{key}")) for key, label in METHODS], rank=True,
                       description="Mean forecasts before allocation; best bold and underlined, second best bold.")

    permutation = evaluation("ordering")
    original = values("window/main/historic_oscar")
    rows = []
    for name in names:
        if name not in permutation:
            continue
        row = permutation[name]
        sr = [r["metrics"]["sharpe"] for r in row["results"]]
        if any(v is None for v in sr):
            low = high = deviation = None
        else:
            low, high = min(sr), max(sr)
            deviation = float(np.std(sr, ddof=1)) if len(sr)>1 else None
        point = original.get(name, {})
        rows.append([name, point.get("sharpe"), point.get("wealth"), row["mean"]["sharpe"],
                     row["mean"]["wealth"], low, high, deviation])
    simple("tab_perm", ["Market", "Original SR", "Original W", "Mean SR", "Mean W", "Min SR", "Max SR", "SD SR"],
           rows, [None] + [".4f"] * 7)

    sign = []
    for position in ["long-only", "long-short"]:
        for kind, optimizer in [("historic","oscar"),("historic","sdp"),("pfl","oscar"),("pfl","sdp"),("dfstpo","dfstpo")]:
            key = "dfstpo" if kind == "dfstpo" else f"{kind}_{optimizer}"
            label = dict(METHODS)[key] + " / " + position
            sign.append((label, values(f"sign/{kind}_{optimizer}_{position}")))
    tables.performance("tab_sign", sign)

    diagnostic = markets_in(read(root / "diagnostics.json"))
    greedy_rows, face_rows, calibration_rows = [], [], []
    for name in names:
        if name not in diagnostic:
            continue
        row = diagnostic[name]
        components = row["components"]
        greedy_rows.append([name, 100*np.mean([c["solver_agreement"] for c in components]),
                            100*np.mean([c["precision_agreement"] for c in components]),
                            100*row["fallback_count"]/row["signals"],
                            float(np.median([v for c in components for v in c["ratio"]]))])
        final = row["final_face"]
        face_rows.append([name, final["kkt_residual"], final["value_upper_ratio"]["median"]])
        c = row["calibration"]
        calibration_rows.append([name, c["before"]["max"], c["after"]["p95"], c["after"]["max"],
                                 100*c["coordinate_box_exceeded"], c["measured_bound_ratio"]["max"]])
    simple("tab_alpha_quality", ["Market","Solver agreement (%)","Precision agreement (%)","Fallback (%)","Median Phi/U"],
           greedy_rows, [None,".1f",".1f",".1f",".4f"])
    simple("tab_final_face_audit", ["Market","Max KKT residual","Median Phi/U"], face_rows, [None,".2e",".4f"])
    simple("tab_calibration_audit", ["Market","Pre max","Post 95%","Post max","Post > 1 (%)","Max bound ratio"],
           calibration_rows, [None] + [".4f"] * 5)

    tables.performance("tab_input_comparison", [(label + " / " + rep, values(f"{rep}/main/{key}"))
                                               for rep in ["raw","window"] for key,label in LEARNED]
                       if (root/"evaluations"/"raw"/"main").exists() else [])
    if (root/"evaluations"/"correction").exists():
        tables.performance("tab_deployment_components", [
            ("Complete", values("window/main/rastpo")),
            ("w.o. correction", values("correction/without_correction")),
            ("Corrected mean only", values("correction/corrected_mean"))])

    bootstrap = []
    if (root/"bootstrap").exists():
        for key, label in LEARNED:
            folder = {"rastpo":"rastpo_alpha", "dfstpo":"dfstpo_dfstpo"}.get(key,key)
            deviation = {}
            for name in names:
                report = read(root/"bootstrap"/folder/name/f"resampling_0_{cfg['bootstrap_replicates']}.json")
                if report is not None:
                    record = markets_in(report).get(name)
                    if record is not None and len(record["results"]) == cfg["bootstrap_replicates"]:
                        deviation[name] = record["standard_deviation"]
            bootstrap.append((label, deviation))
        tables.performance("tab_member_bootstrap", bootstrap,
                           description="Sample standard deviations across complete member-bootstrap replicates; -- means incomplete.")

    def fits(kind, count):
        records = [read(root/"models"/"window"/kind/f"seed_{i}"/"fit.json") for i in range(count)]
        return records if all(r is not None for r in records) else None

    def cost(records, field):
        return None if records is None else sum(r[field]["seconds"] for r in records)

    anchors = fits("pfl", cfg["members"])
    pfl = fits("pfl", cfg["baseline_members"])
    dfstpo = fits("dfstpo", cfg["baseline_members"])
    correction = fits(f"faces_{cfg['faces']}", cfg["members"])
    resource_rows, sdp_costs = [], []
    for key, label in METHODS:
        record = evaluation(f"window/main/{key}")
        if not record:
            continue
        pool = (anchors+correction if anchors is not None and correction is not None else None) if key == "rastpo" else (
            dfstpo if key == "dfstpo" else pfl if key.startswith("pfl_") else [])
        network_count = 2*cfg["members"] if key=="rastpo" else cfg["baseline_members"] if not key.startswith("historic_") else 0
        measured = [r["resources"] for m in record.values() for r in m["results"]]
        memories = measured + [r[f] for r in (pool or []) for f in ["training_resources","prediction_resources"]]
        complete = len(record) == len(names)
        backtest = sum(r["seconds"] for r in measured) if complete else None
        resource_rows.append([label, network_count, cost(pool,"training_resources"), cost(pool,"prediction_resources"),
                              backtest, max(r["gpu_peak_allocated_mib"] for r in memories),
                              max(r["cpu_process_peak_rss_mib"] for r in memories)])
        if key.endswith("_sdp"):
            sdp_costs.append(dict(method=label, solve_seconds=sum(
                (r.get("selection_resources") or {}).get("solve_seconds",0)
                for m in record.values() for r in m["results"])))
    simple("tab_resources_total", ["Method","Networks","Train (s)","Forecast (s)","Backtest (s)","GPU (MiB)","CPU (MiB)"],
           resource_rows, [None,"d",".2f",".2f",".2f",".1f",".1f"],
           description="Training and forecast costs include anchors for RA-STPO. SDP search time is recorded separately in summary.json.")

    benchmark_names = cfg.get("benchmark_markets") or list(dict.fromkeys([names[0],names[-1]]))
    latency_rows = []
    for key, label in METHODS:
        folder = {"rastpo":"rastpo_alpha", "dfstpo":"dfstpo_dfstpo"}.get(key,key)
        record = read(root/"benchmarks"/(folder+".json"))
        if record is None:
            continue
        market_rows = markets_in(record)
        latency = [1000*market_rows[n]["results"][0]["median_seconds"] if n in market_rows else None for n in benchmark_names]
        measurements = [t for m in market_rows.values() for r in m["results"] for t in r["measurements"]]
        device = record["environment"]["device"]
        memory = max(t["gpu_incremental_peak_mib"] if device.startswith("cuda") else t["cpu_process_peak_rss_mib"]
                     for t in measurements)
        latency_rows.append([label, *latency, memory, device])
    simple("tab_resources_decision", ["Method",*[n+" (ms)" for n in benchmark_names],"Peak (MiB)","Device"],
           latency_rows, [None]+[".3f"]*len(benchmark_names)+[".1f",None])

    sensitivity, training_rows = [], []
    for j in cfg["face_counts"]:
        for m in cfg["sensitivity_members"]:
            sensitivity.append((f"J={j}, M={m}", values(f"sensitivity/M{m}_J{j}")))
    tables.performance("tab_mj_performance", sensitivity)
    if (root/"evaluations"/"sensitivity").exists():
        for m in cfg["sensitivity_members"]:
            row = [m]
            for j in cfg["face_counts"]:
                a, c = fits("pfl",m), fits(f"faces_{j}",m)
                row.append(None if a is None or c is None else cost(a,"training_resources")+
                           cost(a,"prediction_resources")+cost(c,"training_resources"))
            training_rows.append(row)
        simple("tab_mj_training", ["M",*[f"J={j}" for j in cfg["face_counts"]]], training_rows,
               ["d"]+[".2f"]*len(cfg["face_counts"]))

    jobs = []
    for path in sorted((root/"logs").glob("*.status")):
        if path.stem != "summarize":
            jobs.append(dict(name=path.stem, status=path.read_text().strip(),
                             log=f"logs/{path.stem}.log"))
    summary = dict(configuration=cfg, tables=tables.index, jobs=jobs, sdp_selection_costs=sdp_costs)
    save_json(root/"summary.json",summary)
    lines = ["# Experiment results", "", "Tables contain measured results from this output directory.", "",
             "Each market has separate SR and W columns. A dash denotes a missing result.", ""]
    lines += [f"- [{t['name']}.csv](tables/{t['name']}.csv) · [TeX](tables/{t['name']}.tex)" for t in tables.index]
    incomplete = [j for j in jobs if j["status"]!="complete"]
    if incomplete:
        lines += ["", "## Incomplete stages", ""] + [f"- {j['name']}: {j['status']} ([log]({j['log']}))" for j in incomplete]
    (root/"RESULTS.md").write_text("\n".join(lines)+"\n")
    print(f"Wrote {len(tables.index)} table pairs to {tables.root}",flush=True)
    return summary
