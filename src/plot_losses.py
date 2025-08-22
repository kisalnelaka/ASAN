import matplotlib.pyplot as plt

for name in ['Baseline', 'MIT-Inspired', 'ASAN']:
    with open(f'results/{name}_losses.txt', 'r') as f:  # Assume first fold for simplicity
        losses = [float(line) for line in f.readlines()]
    plt.plot(losses, label=name)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Loss Comparison (Fold 0)')
plt.savefig('results/loss_plot.png')
plt.show()