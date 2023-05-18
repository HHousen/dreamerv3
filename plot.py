import matplotlib.pyplot as plt
import numpy as np
import json


for number in [16, 64]:
    print(f"Plotting {number}...")
    with open(f"dreamer-boxworld-scale_big_train{number}/metrics.jsonl", "r") as metrics_file:
        metrics = [json.loads(line) for line in metrics_file]
        data = []
        for metric in metrics:
            try:
                data.append(metric["episode/score"])
            except KeyError:
                pass
        data = np.convolve(np.array(data) >= 10, np.ones((500,))/500, mode='valid')
        plt.plot(data)
        plt.savefig(f"plot{number}.png")