import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="LNP Inverse Problem Pipeline")
    parser.add_argument("--generate", action="store_true",
                        help="Generate dataset")
    parser.add_argument("--train", action="store_true",
                        help="Train all models")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate all models")
    parser.add_argument("--all", action="store_true",
                        help="Run full pipeline: generate + train + evaluate")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    args = parser.parse_args()

    if args.all or args.generate:
        print("=== Generating Dataset ===")
        subprocess.run([sys.executable, "simulate.py"])

    if args.all or args.train:
        print("=== Training Models ===")
        for model in ["mlp", "cnn", "pinn"]:
            print(f"\nTraining {model.upper()}...")
            subprocess.run([sys.executable, "train.py",
                          "--model", model,
                          "--epochs", str(args.epochs)])

    if args.all or args.evaluate:
        print("=== Evaluating Models ===")
        subprocess.run([sys.executable, "evaluate.py"])


if __name__ == "__main__":
    main()