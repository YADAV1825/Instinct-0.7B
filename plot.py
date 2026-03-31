import matplotlib.pyplot as plt
import os

def create_training_graphs(log_filepath, ppl_filepath, output_filename="training_curves.png"):
    """
    Reads training logs and perplexity logs to generate three separate graphs:
    1. Training Loss
    2. Validation Loss
    3. Validation Perplexity
    """
    tokens_loss = []
    train_losses = []
    val_losses = []

    # Read and process the training log
    if os.path.exists(log_filepath):
        with open(log_filepath, 'r') as file:
            for line in file:
                parts = line.strip().split('|')
                if len(parts) == 3:
                    tokens = int(parts[0].replace('tokens', '').strip())
                    t_loss = float(parts[1].split('=')[1].strip())
                    v_loss = float(parts[2].split('=')[1].strip())

                    # Convert tokens to billions
                    tokens_loss.append(tokens / 1_000_000_000)
                    train_losses.append(t_loss)
                    val_losses.append(v_loss)
    else:
        print(f"File not found: {log_filepath}")
        return

    tokens_ppl = []
    val_ppl = []

    # Read and process the perplexity log
    if os.path.exists(ppl_filepath):
        with open(ppl_filepath, 'r') as file:
            for line in file:
                parts = line.strip().split('|')
                if len(parts) == 2:
                    tokens = int(parts[0].replace('tokens', '').strip())
                    ppl = float(parts[1].split('=')[1].strip())

                    # Convert tokens to billions
                    tokens_ppl.append(tokens / 1_000_000_000)
                    val_ppl.append(ppl)
    else:
        print(f"File not found: {ppl_filepath}")
        return

    # Set up the three distinct graphs
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

    # 1. Training Loss Graph
    ax1.plot(tokens_loss, train_losses, color='blue', linewidth=1.5)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Tokens (Billions)')
    ax1.set_ylabel('Loss')
    ax1.grid(True, linestyle='--', alpha=0.7)

    # 2. Validation Loss Graph
    ax2.plot(tokens_loss, val_losses, color='orange', linewidth=1.5)
    ax2.set_title('Validation Loss')
    ax2.set_xlabel('Tokens (Billions)')
    ax2.set_ylabel('Loss')
    ax2.grid(True, linestyle='--', alpha=0.7)

    # 3. Validation Perplexity Graph
    ax3.plot(tokens_ppl, val_ppl, color='green', linewidth=1.5)
    ax3.set_title('Validation Perplexity')
    ax3.set_xlabel('Tokens (Billions)')
    ax3.set_ylabel('Perplexity')
    ax3.grid(True, linestyle='--', alpha=0.7)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_filename, dpi=600)
    print(f"Graphs successfully created and saved as {output_filename}")

if __name__ == "__main__":
    create_training_graphs("training_log.txt", "val_perplexity.txt")