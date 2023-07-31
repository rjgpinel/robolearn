import json
from pathlib import Path
import numpy as np
import yaml
import matplotlib.pyplot as plt
import click
from collections import OrderedDict

from robolearn.utils.lines import Lines


def scale_logs(d, key, scale):
    for k, v in d.items():
        if key in k and (isinstance(v, np.ndarray) or isinstance(v, float)):
            d[k] = scale * v
        elif isinstance(v, dict):
            d[k] = scale_logs(v, key, scale)
        elif isinstance(v, list):
            d[k] = [scale_logs(x, key, scale) for x in v]
    return d


def plot_logs(logs, x_key, y_key, size, vmin, vmax, last_log_idx=None, smooth=0.0):
    domains = []
    lines = []
    labels = []
    y_keys = y_key.split("/")
    for name, log in logs.items():
        logs[name] = log[:last_log_idx]
    for name, log in logs.items():
        domain = [x[x_key] for x in log if y_keys[0] in x]
        log_plot = [x[y_keys[0]] for x in log if y_keys[0] in x]
        if not log_plot:
            continue
        for y_key in y_keys[1:]:
            if y_key in log_plot[0]:
                log_plot = [x[y_key] for x in log_plot if y_key in x]
        log_plot = [
            x if not np.isnan(x) else log[i - 1] for i, x in enumerate(log_plot)
        ]
        domains.append(domain)
        lines.append(np.array(log_plot)[:, None])
        labels.append(name)

    m = min([np.min(l) for l in lines])
    M = max([np.max(l) for l in lines])

    if vmin is not None:
        m = vmin
    if vmax is not None:
        M = vmax
    delta = 0.1 * (M - m)

    ratio = 0.6
    figsizes = {"tight": (8, 6), "large": (16 * ratio, 10 * ratio)}
    figsize = figsizes[size]

    # plot parameters
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    plot_lines = Lines(resolution=50, smooth=smooth)
    plot_lines.LEGEND["loc"] = "upper left"
    # plot_lines.LEGEND["fontsize"] = "large"
    plot_lines.LEGEND["bbox_to_anchor"] = (0.75, 0.2)
    colors = plot_lines(ax, domains, lines, labels)
    ax.grid(True, alpha=0.5)
    ax.set_ylim(m - delta, M + delta)

    # for (name, log), color in zip(logs.items(), colors):
    #     last_log = log[-1]
    #     x, y = last_log[x_key], last_log[y_key]
    #     ax.scatter(x, y, c=color, marker="x")
    #     ax.annotate(f"{y:.2f}", xy=(x, y), xytext=(x, y + 0.1 * delta))
    plt.show()
    fig.savefig(
        "plot.png", bbox_inches="tight", pad_inches=0.1, transparent=False, dpi=300
    )
    plt.close(fig)


def print_logs(logs, x_key, y_key, last_log_idx=None):
    delim = "   "
    s = ""
    keys = []
    y_keys = y_key.split("/")
    for name, log in logs.items():
        key_in_log = False
        log_idx = last_log_idx
        if log_idx is None:
            log_idx = len(log)
        while not key_in_log:
            log_idx -= 1
            if log_idx < 0:
                break
            key_in_log = y_keys[0] in log[log_idx]
        if not key_in_log:
            continue
        last_log = log[log_idx]
        log_x = last_log[x_key]
        log_y = last_log[y_keys[0]]
        for y_key in y_keys[1:]:
            log_y = log_y[y_key]
        s += f"{name}:\n"
        # s += f"{delim}{x_key}: {log_x}\n"
        s += f"{delim}{y_key}: {log_y:.4f}\n"
        keys += list(last_log.keys())
    keys = sorted(list(set(keys)))
    keys = ", ".join(keys)
    s = f"keys: {keys}\n" + s
    print(s)


def read_logs(root, logs_path, filename, filt_key):
    logs = {}
    for name, path in logs_path.items():
        path = root / path
        matches = path.rglob(filename)
        if not matches:
            print(f"Skipping {name} that has no log file")
            continue
        matches = sorted(matches)
        for match in matches:
            if filt_key and filt_key not in str(match):
                continue
            match_name = f"{name} - {match.parent.name}"
            logs[match_name] = []
            with open(match, "r") as f:
                for line in f.readlines():
                    d = json.loads(line)
                    logs[match_name].append(d)
    return logs


@click.command()
@click.argument("log_path", type=str)
@click.option("--filename", default="log.txt", type=str)
@click.option("--x-key", default="num_updates", type=str)
@click.option("--y-key", default="val_loss_vel", type=str)
@click.option("--filt-key", default=None, type=str)
@click.option("-s", "--size", default="large", type=str)
@click.option("--vmin", default=None, type=float)
@click.option("--vmax", default=None, type=float)
@click.option("-plot", "--plot/--no-plot", default=True, is_flag=True)
@click.option("-smooth", "--smooth", default=0.0)
def main(log_path, filename, x_key, y_key, filt_key, size, vmin, vmax, plot, smooth):
    abs_path = Path(__file__).parent / log_path
    if abs_path.exists():
        log_path = abs_path
    config = yaml.load(open(log_path, "r"), Loader=yaml.FullLoader)
    root = Path(config["root"])
    logs_path = OrderedDict(reversed(list(config["logs"].items())))
    if vmin is None:
        vmin = config.get("vmin", None)
    if vmax is None:
        vmax = config.get("vmax", None)
    last_log_idx = config.get(y_key, None)

    logs = read_logs(root, logs_path, filename, filt_key)
    if not logs:
        print(f"Empty logs path: {logs_path}")
        return

    # print(f"keys: {list(next(iter(logs.values()))[0].keys())}")
    # logs = scale_logs(logs, "iou", 100)

    print_logs(logs, x_key, y_key, last_log_idx)
    if plot:
        plot_logs(logs, x_key, y_key, size, vmin, vmax, last_log_idx, smooth)


if __name__ == "__main__":
    main()
