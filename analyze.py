import numpy as np
import matplotlib.pyplot as plt

# Philip Mocz (2025)
# Flatiron Institute


def main():
    # load log files and plot scaling
    methods = ["jfft", "xfft"]
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
    plt.figure()
    timings = np.array(timings).reshape(
        len(methods), len(precisions), len(n_devices), len(resolutions)
    )
    for j, method in enumerate(methods):
        for k, precision in enumerate(precisions):
            for i, n_dev in enumerate(n_devices):
                timings_slice = timings[j, k, i, :]
                if np.all(np.isnan(timings_slice)):
                    continue
                plt.plot(
                    resolutions,
                    timings_slice,
                    marker="o",
                    label=f"{method} ({precision}) #gpus={n_dev}",
                )
    plt.xscale("log")
    plt.yscale("log")
    plt.xticks(resolutions, labels=[str(r) for r in resolutions])
    times_ms = [0.1, 1, 10, 100]
    plt.yticks(times_ms, labels=[str(r) for r in times_ms])
    plt.xlabel("resolution")
    plt.ylabel("time (ms) --  lower is better")
    plt.title("Scaling of 3D FFT")
    plt.legend()
    plt.savefig("scaling.png")


if __name__ == "__main__":
    main()
