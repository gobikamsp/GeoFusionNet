import os
import matplotlib.pyplot as plt

LOG_FILE = "logs/train_loss.txt"
OUT_FILE = "logs/loss_curve.png"

def main():
    if not os.path.exists(LOG_FILE):
        print(f"Log file not found: {LOG_FILE}")
        return

    epochs, losses = [], []

    with open(LOG_FILE, "r") as f:
        for line in f:
            epoch, loss = line.strip().split(",")
            epochs.append(int(epoch))
            losses.append(float(loss))

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Average Training Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.tight_layout()

    os.makedirs("logs", exist_ok=True)
    plt.savefig(OUT_FILE)
    plt.close()

    print(f"Loss curve saved to {OUT_FILE}")

if __name__ == "__main__":
    main()
