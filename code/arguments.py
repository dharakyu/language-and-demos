from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

def get_args():
    parser = ArgumentParser(
        description='Train',
        formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--num_batches_per_epoch', default=100, type=int)
    parser.add_argument('--log_interval', default=100, type=int)
    parser.add_argument('--lr', default=1e-5, type=float)


    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument(
        "--wandb_project_name", default="signaling-bandits", help="wandb project name"
    )
    parser.add_argument('--name', default=None)
    parser.add_argument('--group', default=None)

    parser.add_argument('--num_reward_matrices', default=36, type=int)
    parser.add_argument('--embedding_size', default=64, type=int)
    parser.add_argument('--hidden_size', default=100, type=int)

    parser.add_argument('--chain_length', default=2, type=int)
    parser.add_argument('--num_listener_views', default=1, type=int)

    parser.add_argument('--partial_reward_matrix', action='store_true')
    parser.add_argument('--use_same_agent', action='store_true')    # option to use the same agent for all the agents in the chain

    args = parser.parse_args()
    return args