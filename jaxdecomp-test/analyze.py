import numpy as np
import matplotlib.pyplot as plt

# Philip Mocz (2025)
# Flatiron Institute


def main():
    # load log files and plot scaling
    methods = ["jfft", "jdfft"]  # "xfft"
    resolutions = [64, 128, 256, 512, 1024, 2048]
    n_devices = [1, 2, 4, 8, 16]
    precisions = ["single", "double"]
    timings = []
    for method in methods:
        for precision in precisions:
            for n_dev in n_devices:
                for res in resolutions:
                    log_filename = f"log_{method}_n{res}_d{n_dev}_{precision}.txt"
                    try:
                        with open(log_filename, "r") as log_file:
                            timing = float(log_file.readline().strip())
                            timings.append(timing)
                    except FileNotFoundError:
                        print(f"Log file {log_filename} not found.")
                        timings.append(np.nan)

    # Plot the results
    for k, precision in enumerate(precisions):
        plt.figure()
        timings = np.array(timings).reshape(
            len(methods), len(precisions), len(n_devices), len(resolutions)
        )
        for j, method in enumerate(methods):
            for i, n_dev in enumerate(n_devices):
                timings_slice = timings[j, k, i, :]
                if np.all(np.isnan(timings_slice)):
                    continue
                plt.plot(
                    resolutions,
                    timings_slice,
                    marker="o" if j == 0 else "s",
                    linewidth=0.5 if j == 0 else 1,
                    label=f"{method} #gpus={n_dev}",
                )
                if not np.isnan(timings_slice[-1]):
                    plt.text(
                        resolutions[-1],
                        timings_slice[-1] * 0.55,
                        f"{timings_slice[-1]:.1f}",
                        fontsize=8,
                        ha="center",
                        va="bottom",
                    )

        plt.xscale("log")
        plt.yscale("log")
        plt.xticks(resolutions, labels=[str(r) for r in resolutions])
        times_ms = [0.01, 0.1, 1, 10, 100, 1000]
        plt.yticks(times_ms, labels=[str(r) for r in times_ms])
        plt.xlabel("resolution")
        plt.ylabel("time (ms) -  lower is better")
        plt.ylim(0.05, 4000.0)
        plt.title(f"Scaling of 3D FFT ({precision})")
        plt.legend()
        plt.savefig(f"scaling_{precision}.png")


if __name__ == "__main__":
    main()
