import matplotlib.pyplot as plt
import numpy as np
import json

with open("dreamer-boxworld-logs/metrics.jsonl", "r") as metrics_file:
    metrics = [json.loads(line) for line in metrics_file]
    data = []
    for metric in metrics:
        try:
            data.append(metric["episode/score"])
        except KeyError:
            pass
    data = np.convolve(data, np.ones((500,))/500, mode='valid')
    plt.plot(data)
    plt.savefig("plot.png")