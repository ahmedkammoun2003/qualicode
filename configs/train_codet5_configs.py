
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Training CodeT5 model for code generation")
    parser.add_argument('--model', default='codet5-large-ntp-py', type=str, help='type of transformers model as model backbone')
    parser.add_argument('--model-path', default=None, type=str, help='path to model backbone pretrained weights') 
    parser.add_argument('--save-dir', default='', type=str, help='path to save trained model checkpoints') 

    # Dataloading
    parser.add_argument('--train-path', default='', type=str, help='path to training data')
    parser.add_argument('--sample-mode', default='uniform_sol', help='sampling output programs following a uniform distribution by program population')

    # Model
    parser.add_argument('--tuning-mode', default='plan', type=str, help='tuning mode for training LMs')
    parser.add_argument('--clone-pl-head', default=True, action='store_true', help='Optional: clone a seperate linear layer for plan generation')

    # Training
    parser.add_argument('--epochs', default=20, type=int, help='total number of training epochs')
    parser.add_argument('--lr', default=2e-5, type=float, help='training learning rate')
    parser.add_argument('--batch-size-per-replica', default=2, type=int, help='batch size per GPU')
    parser.add_argument('--grad-acc-steps', default=16, type=int, help='number of training steps before each gradient update')
    parser.add_argument('--deepspeed', default=None, type=str, help='path to deepspeed configuration file; set None if not using deepspeed')
    parser.add_argument('--fp16', default=True, action='store_true', help='set 16-bit training to reduce memory usage')
    # Update local-rank argument to be optional and default to -1
    parser.add_argument('--local-rank', type=int, default=-1,
                    help='Local rank passed from distributed launcher')

    args = parser.parse_args()
    return args


