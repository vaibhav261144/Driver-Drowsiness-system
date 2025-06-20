import os
import argparse
from drowsiness_model import train_eye_model

def get_args():
    parser = argparse.ArgumentParser(description="Train Eye State Classification Model")
    parser.add_argument("--train_dir", type=str, default="data/train",
                        help="Directory containing training data")
    parser.add_argument("--val_dir", type=str, default="data/validation",
                        help="Directory containing validation data")
    parser.add_argument("--output_model", type=str, default="models/eye_state_model.h5",
                        help="Path to save the trained model")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of epochs to train")
    return parser.parse_args()

def main():
    args = get_args()
    
    # Check if directories exist
    if not os.path.exists(args.train_dir):
        print(f"Training directory {args.train_dir} does not exist.")
        print("Please create the directory with 'open' and 'closed' subdirectories containing eye images.")
        return
        
    if not os.path.exists(args.val_dir):
        print(f"Validation directory {args.val_dir} does not exist.")
        print("Please create the directory with 'open' and 'closed' subdirectories containing eye images.")
        return
    
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    
    print("Starting model training...")
    model = train_eye_model(
        train_data_dir=args.train_dir,
        validation_data_dir=args.val_dir,
        model_save_path=args.output_model,
        batch_size=args.batch_size,
        epochs=args.epochs
    )
    
    print(f"Model trained and saved to {args.output_model}")

if __name__ == "__main__":
    main() 