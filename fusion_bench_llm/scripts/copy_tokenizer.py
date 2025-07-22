from transformers import AutoTokenizer

import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Copy tokenizer from one model to another."
    )
    parser.add_argument(
        "-i", type=str, required=True, help="Source model name or path."
    )
    parser.add_argument(
        "-o", type=str, required=True, help="Target model name or path."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    source_tokenizer = AutoTokenizer.from_pretrained(args.i)
    source_tokenizer.save_pretrained(args.o)
    print(f"Tokenizer copied from {args.i} to {args.o}")


if __name__ == "__main__":
    main()
