import json
from pathlib import Path
import numpy as np
import yaml
import matplotlib.pyplot as plt
import click
from collections import OrderedDict

from robolearn.utils.lines import Lines

Y_LABEL = {"val_success_rate": "Success Rate", "val_real0_vel_error": "Velocity Error"}


def plot_confidence_interval(
    x, values, z=1.96, color="#2187bb", horizontal_line_width=0.25
):
    mean = np.mean(values)
    stdev = np.std(values)
    confidence_interval = z * stdev / np.sqrt(len(values))

    left = x - horizontal_line_width / 2
    top = mean - confidence_interval
    right = x + horizontal_line_width / 2
    bottom = mean + confidence_interval
    return mean, stdev, confidence_interval


def print_logs(logs, y_key, last_log_idx=None):
    delim = "   "
    s = ""
    keys = []
    y_keys = y_key.split("/")
    last_exp_info = dict()
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
        log_y = last_log[y_keys[0]]
        for y_key in y_keys[1:]:
            log_y = log_y[y_key]
        s += f"{name}:\n"
        s += f"{delim}{y_key}: {log_y:.4f}\n"
        keys += list(last_log.keys())
        exp_v = float(name.split("-")[0].strip())
        last_exp_info[exp_v] = log_y
    keys = sorted(list(set(keys)))
    keys = ", ".join(keys)
    s = f"keys: {keys}\n" + s
    print(s)
    return last_exp_info


def read_logs(root, logs_path, filename, filt_key=None):
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
@click.option("--key", default="cam", type=str)
@click.option("--y-key", default="val_loss_vel", type=str)
@click.option("--vmin", default=None, type=float)
@click.option("--vmax", default=None, type=float)
@click.option("--output_dir", default="./", type=str)
def main(log_path, filename, key, y_key, vmin, vmax, output_dir):
    output_dir = Path(output_dir)
    abs_path = Path(__file__).parent / log_path
    if abs_path.exists():
        log_path = abs_path

    var_factors = []
    if key == "xyz_cam":
        # var_factors = {"xyz_cam_rand_factor": np.arange(0.0, 11.0, 1.0).tolist()}
        var_factors = {
            "xyz_cam_rand_factor": [
                0.0,
                0.125,
                0.25,
                0.375,
                # 0.5,
                0.625,
                0.75,
                0.875,
                1.0,
            ]
        }
        var_value = 0.025
    elif key == "rpy_cam":
        var_factors = {"rpy_cam_rand_factor": np.arange(0.0, 11.0, 1.0).tolist()}
        var_value = 0.025
    elif key == "fov_cam":
        var_factors = {"fov_cam_rand_factor": np.arange(0.0, 11.0, 1.0).tolist()}
        var_value = 1.0
    exp_cfg = {
        "light_rand_factor": 2.0,
        "xyz_cam_rand_factor": 4.0,
        "rpy_cam_rand_factor": 2.0,
        "fov_cam_rand_factor": 1.0,
        "hsv_rand_factor": 0.5,
        "light_pos_rand_factor": 5.0,
        "num_textures": 15000,
    }

    x_label = {
        "hsv": "Color randomization factor",
        "xyz_cam": "XYZ camera randomization (cm)",
        "rpy_cam": "RPY camera randomization (rad)",
        "fov_cam": "FOV camera randomization (º)",
        "textures": "Number of textures",
        "light": "Light randomization factor",
        "light_pos": "Light position randomization factor",
    }

    performances = {}

    log_path = Path(log_path)
    print(log_path)
    for factor_name, var_factor in var_factors.items():
        print(f"Factor name: {factor_name}")
        if factor_name not in performances.keys():
            performances[factor_name] = {}
        for factor in var_factor:
            exp_cfg["factor"] = factor
            exp_cfg[factor_name] = factor
            exp_cfg["factor_name"] = factor_name
            if factor not in performances.keys():
                performances[factor_name][factor] = []
            for i in range(5):
                seed_path = log_path / f"{i}"
                if not seed_path.exists():
                    continue
                factor_path = (
                    seed_path
                    / "lf{light_rand_factor}-xyzf{xyz_cam_rand_factor}-rpyf{rpy_cam_rand_factor}-fovf{fov_cam_rand_factor}-hf{hsv_rand_factor}-nt{num_textures}_{factor_name}_{factor}".format(
                        **exp_cfg
                    )
                )

                print(factor_path)

                if not factor_path.exists():
                    continue
                logs = read_logs(Path("./"), {"": factor_path}, filename)
                if not logs:
                    print(f"Empty logs path: {logs_path}")
                    continue

                for v in reversed(list(logs.values())[0]):
                    if y_key in v.keys():
                        performances[factor_name][factor].append(v[y_key])
                        break

        print(performances)
        means = []
        stdevs = []
        tick = 0

        for factor_name, factor_performances in performances.items():
            for factor, performance in factor_performances.items():
                mean, stdev, _ = plot_confidence_interval(tick, performance)
                tick += 1
                means.append(mean)
                stdevs.append(stdev)

        fig, ax = plt.subplots()

        x_values = [f * var_value for f in factor_performances.keys()]
        ax.plot(
            x_values,
            means,
        )
        ax.fill_between(
            x_values,
            np.array(means) - np.array(stdevs),
            np.array(means) + np.array(stdevs),
            alpha=0.2,
        )
        plt.xlabel(x_label[key])
        plt.ylabel(Y_LABEL[y_key])
    # ax.set_ylim([vmin, vmax])
    # plt.legend()
    output_plot_dir = output_dir / f"{factor_name}.png"
    plt.savefig(f"{str(output_plot_dir)}")


if __name__ == "__main__":
    main()
