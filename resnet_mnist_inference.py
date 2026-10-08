import argparse

import torch
from datasets import load_dataset
from transformers import AutoImageProcessor, ResNetForImageClassification
from tqdm import tqdm


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrained ResNet inference on MNIST")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional: only run on the first N test images (useful for a quick smoke test).",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_name = "microsoft/resnet-50"

    # Load a pretrained ResNet-50 (ImageNet, 1000 output classes) and its image
    # processor, which performs the resize and normalization steps for us.
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = ResNetForImageClassification.from_pretrained(model_name)
    model.to(device).eval()

    # Load the MNIST test split: 28x28 single-channel (grayscale) digit images,
    # 10 classes (0-9).
    test = load_dataset("mnist", split="test")
    if args.max_samples is not None:
        test = test.select(range(min(args.max_samples, len(test))))

    # MNIST images are grayscale ("L" mode), but ResNet expects 3-channel RGB
    # input, so convert each image to RGB first. The processor below then
    # resizes them to ResNet's expected input size (224x224) and normalizes
    # with ImageNet statistics.
    images = [img.convert("RGB") for img in test["image"]]
    labels = test["label"]

    correct = 0
    total = 0

    for start in tqdm(range(0, len(images), args.batch_size), desc="Inference"):
        batch_images = images[start : start + args.batch_size]
        batch_labels = labels[start : start + args.batch_size]

        # Resize + normalize the batch, then run the model.
        inputs = processor(images=batch_images, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        preds = logits.argmax(dim=-1).cpu()

        correct += (preds == torch.tensor(batch_labels)).sum().item()
        total += len(batch_labels)

    accuracy = correct / total
    print(f"\nACCURACY: {accuracy:.4f}  ({correct}/{total} correct)")
    print("Put this number in your commit message, e.g.:")
    print(
        f'  git commit -m "ResNet inference on MNIST (resize images to model input), '
        f'accuracy={accuracy:.4f}"'
    )


if __name__ == "__main__":
    main()
