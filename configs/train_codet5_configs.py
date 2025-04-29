
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Training CodeT5 model for code generation")
    # Existing model arguments
    parser.add_argument('--model', default='codet5-large-ntp-py', type=str, help='type of transformers model as model backbone')
    parser.add_argument('--model-path', default=None, type=str, help='path to model backbone pretrained weights') 
    parser.add_argument('--save-dir', default='', type=str, help='path to save trained model checkpoints') 

    # Dataloading
    parser.add_argument('--train-path', default='', type=str, help='path to training data')
    parser.add_argument('--sample-mode', default='uniform_sol', help='sampling output programs following a uniform distribution by program population')
    
    # Dataset split configurations
    parser.add_argument('--train-split', default=0.7, type=float, help='proportion of data for training')
    parser.add_argument('--val-split', default=0.2, type=float, help='proportion of data for validation')
    parser.add_argument('--test-split', default=0.1, type=float, help='proportion of data for testing')

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
    parser.add_argument('--local-rank', type=int, default=-1, help='Local rank passed from distributed launcher')

    # Evaluation configurations
    parser.add_argument('--eval-batch-size', default=8, type=int, help='batch size for evaluation')
    parser.add_argument('--eval-frequency', default=500, type=int, help='evaluate model every N steps')
    parser.add_argument('--save-freq', default=500, type=int, help='save model every N steps')
    parser.add_argument('--save-total-limit', default=5, type=int, help='maximum number of checkpoints to keep')
    parser.add_argument('--log-freq', default=100, type=int, help='log training metrics every N steps')

    # Debug mode
    parser.add_argument('--db', action='store_true', help='run in debug mode with limited samples')

    args = parser.parse_args()
    return args

# Remove any global variables or direct args assignment


