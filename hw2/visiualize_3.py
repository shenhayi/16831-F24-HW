import json
import matplotlib.pyplot as plt
import os

# List of JSON files to read
json_file_paths = [
    '/home/haoyus/16831/16831-F24-HW/hw2/result/json/q1_lb_no_rtg_dsa_CartPole-v0_02-10-2024_00-06-38.json',
    '/home/haoyus/16831/16831-F24-HW/hw2/result/json/q1_lb_rtg_dsa_CartPole-v0_02-10-2024_00-11-12.json',
    '/home/haoyus/16831/16831-F24-HW/hw2/result/json/q1_lb_rtg_na_CartPole-v0_02-10-2024_00-15-38.json'
]

# Optional: Labels for each dataset (adjust as needed)
labels = [
    'Run 1 no rtg+dsa',
    'Run 2 rtg+dsa',
    'Run 3 rtg+na'
]

# Colors or line styles for each dataset (optional)
line_styles = ['-', '--', '-.']
colors = ['blue', 'green', 'red']

# Initialize the plot
plt.figure(figsize=(12, 8))

# Loop over each JSON file
for idx, json_file_path in enumerate(json_file_paths):
    # Check if the file exists
    if not os.path.isfile(json_file_path):
        print(f"File {json_file_path} not found. Skipping.")
        continue

    # Load data from the JSON file
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error decoding JSON in file {json_file_path}. Skipping.")
        continue

    iterations = []
    average_returns = []

    # Extract iterations and average returns from the data
    for entry in data:
        if isinstance(entry, list) and len(entry) >= 3:
            # entry is a list: [timestamp, iteration, average_return]
            timestamp = entry[0]
            iteration = entry[1]
            avg_return = entry[2]

            # Append the iteration and average return to the lists
            iterations.append(iteration)
            average_returns.append(avg_return)
        else:
            print(f"Unexpected entry format: {entry}")

    # Sort the data by iteration in case it's unordered
    iterations, average_returns = zip(*sorted(zip(iterations, average_returns)))

    # Limit the iterations to the desired range (0-100)
    max_iteration = 100
    iterations = list(iterations)
    average_returns = list(average_returns)
    if max(iterations) > max_iteration:
        # Find the index where iterations exceed max_iteration
        cutoff_index = next((i for i, x in enumerate(iterations) if x > max_iteration), len(iterations))
        iterations = iterations[:cutoff_index]
        average_returns = average_returns[:cutoff_index]

    # Plot the data
    plt.plot(
        iterations,
        average_returns,
        label=labels[idx] if idx < len(labels) else f'Dataset {idx+1}',
        linestyle=line_styles[idx % len(line_styles)],
        color=colors[idx % len(colors)]
    )

# Set plot parameters
plt.xlabel('Iteration')
plt.ylabel('Average Return')
plt.title('Cartpole Large Batch')
plt.grid(True)
plt.legend()
plt.xlim(0, 100)
plt.ylim(0, 300)
plt.tight_layout()

# Show the plot
plt.show()
