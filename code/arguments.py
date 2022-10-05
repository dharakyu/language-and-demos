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
    parser.add_argument('--group', default=None)

    parser.add_argument('--num_colors', default=4, type=int)
    parser.add_argument('--num_shapes', default=4, type=int)

    parser.add_argument('--embedding_size', default=64, type=int)
    parser.add_argument('--hidden_size', default=100, type=int)

    parser.add_argument('--discrete_comm', action='store_true')
    parser.add_argument('--max_message_len', default=4, type=int)
    parser.add_argument('--vocab_size', default=80, type=int)

    parser.add_argument('--chain_length', default=2, type=int)

    parser.add_argument('--partial_reward_matrix', action='store_true')
    parser.add_argument('--use_same_agent', action='store_true')    # option to use the same agent for all the agents in the chain
    parser.add_argument('--train_chain_length', default=None, type=int)
    parser.add_argument('--optimize_jointly', action='store_true')

    parser.add_argument('--chunks', nargs='+', default=None, type=int)
    parser.add_argument('--num_utilities_seen_in_training', nargs='+', default=None, type=int)

    parser.add_argument('--ingest_multiple_messages', action='store_true')

    parser.add_argument('--shuffle_agents', action='store_true')

    parser.add_argument('--same_agent_view', action='store_true')   # whether each agent gets the same view of the objects
    parser.add_argument('--no_additional_info', action='store_true')    # whether there is no additional information provided after agent 0

    parser.add_argument('--learn_from_demos', action='store_true')
    parser.add_argument('--num_examples_for_demos', default=10, type=int)

    parser.add_argument('--inductive_bias', action='store_true')

    parser.add_argument('--save_dir', default='/home/dharakyu/signaling-bandits/outputs', type=str)
    parser.add_argument('--save_outputs', action='store_true')

    args = parser.parse_args()
    return args