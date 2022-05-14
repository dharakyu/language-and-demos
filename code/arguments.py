from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

def get_args():
    parser = ArgumentParser(
        description='Train',
        formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=40, type=int)
    parser.add_argument('--num_batches_per_epoch', default=100, type=int)
    parser.add_argument('--log_interval', default=100, type=int)
    parser.add_argument('--lr', default=1e-5, type=float)


    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument(
        "--wandb_project_name", default="signaling-bandits", help="wandb project name"
    )
    parser.add_argument('--name', default=None)

    parser.add_argument('--num_reward_matrices', default=36, type=int)

    args = parser.parse_args()
    return args