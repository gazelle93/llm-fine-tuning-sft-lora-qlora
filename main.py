import argparse
import data
import model as model_utils
import train
import eval
from config import Config

def main(args):
    print(f" --- Starting with model: {args.model_name}")
    
    # Update config dynamically
    Config.MODEL_NAME = args.model_name
    Config.EPOCHS = args.epochs
    Config.BATCH_SIZE = args.batch_size
    Config.LEARNING_RATE = args.learning_rate
    Config.OUTPUT_DIR = args.output_dir
    Config.MAX_OUTPUT_TOKEN = args.max_output_token
    Config.TEMPERATURE = args.temperature
    Config.TOP_P = args.top_p
    Config.MAX_LENGTH = args.max_length

    print(" --- Loading tokenizer...")
    tokenizer = model_utils.load_tokenizer()

    if args.fine_tuning == 'sft':
        print(" --- Loading model with SFT...")
        model = model_utils.load_model_with_sft()
    elif args.fine_tuning == 'lora':
        print(" --- Loading model with LoRA...")
        model = model_utils.load_model_with_lora()
    elif args.fine_tuning == 'qlora':
        print(" --- Loading model with QLoRA...")
        model = model_utils.load_model_with_qlora()

    dataset = data.get_dataset()

    print(" --- Starting training...")
    train.train(model, tokenizer, dataset)

    print(" --- Evaluating model...")
    eval.evaluate_model(model, tokenizer)

    print("\n --- All done!")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Qwen with LoRA + DPO")

    parser.add_argument(
        "--model_name",
        type=str,
        default=Config.MODEL_NAME,
        help="Hugging Face model ID (default: Qwen/Qwen1.5-1.8B)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=Config.EPOCHS,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=Config.BATCH_SIZE,
        help="Per-device batch size"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=Config.LEARNING_RATE,
        help="Learning rate"
    )
    parser.add_argument(
        "--max_output_token",
        type=int,
        default=Config.MAX_OUTPUT_TOKEN,
        help="Maximum number of new tokens to generate during evaluation"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=Config.TOP_P,
        help="Nucleus sampling probability threshold (controls diversity during generation)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=Config.TEMPERATURE,
        help="Sampling temperature (higher = more randomness in generation)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=Config.MAX_LENGTH,
        help="Maximum sequence length for tokenizing prompts and completions (input will be truncated to this length)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=Config.OUTPUT_DIR,
        help="Directory to save model checkpoints"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation after training"
    )
    parser.add_argument(
        "--fine_tuning",
        type=str,
        choices=["sft", "lora", "qlora"],
        default="lora",
        help="Type of fine-tuning to run: 'sft' (standard full-parameter), 'lora' (LoRA adapters), or 'qlora' (quantized LoRA)."
    )

    args = parser.parse_args()
    main(args)
