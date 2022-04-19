"""
io util
"""


from .games import GAMES
import yaml


def load_config(fname):
    with open(fname, "r") as f:
        config = yaml.safe_load(f)
    return config


def parse_args(defaults=False):
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description="Train", formatter_class=ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--game_config",
        default="./configs/simple_competitive.yaml",
        help="path to game config file",
    )
    parser.add_argument(
        "--mode",
        default="rf",
        choices=["rf", "ex"],
        help="training mode. rf = reinforce; ex = exhaustive enumeration",
    )
    parser.add_argument(
        "--max_dialog_rounds",
        default=1,
        type=int,
        help="Maximum number of rounds of dialog",
    )
    parser.add_argument(
        "--vocab_size",
        default=20,
        type=int,
        help="Communication vocab size (default is number of shapeworld shapes + colors)",
    )
    parser.add_argument(
        "--continuous_dialog",
        action="store_true",
        help="Use continuous communication",
    )
    parser.add_argument(
        "--no_masking",
        action="store_true",
        help="Don't do masking",
    )
    parser.add_argument(
        "--max_lang_length",
        default=4,
        type=int,
        help="NOT IMPLEMENTED YET: Maximum language length, including SOS and EOS",
    )
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--clip", default=100.0, type=float, help="Gradient clipping")
    parser.add_argument("--num_epochs", default=200, type=int)
    parser.add_argument("--num_steps_per_epoch", default=100, type=int)
    parser.add_argument("--tau", default=1.0, type=float, help="Gumbel-softmax tau")
    parser.add_argument(
        "--save_interval",
        default=10,
        type=int,
        help="How often (in epochs) to save lang + model",
    )
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--softmax_temp", default=1.0, type=float)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--wandb", action="store_true", help="log with wandb")
    parser.add_argument(
        "--wandb_project_name", default="negotiation-rl", help="wandb project name"
    )
    parser.add_argument("--n_workers", default=0, type=int)
    parser.add_argument("--name", default=None)
    parser.add_argument("--debug", action="store_true")

    if defaults:
        args = parser.parse_args([])
    else:  # From CLI
        args = parser.parse_args()

    # Load config
    game_config = load_config(args.game_config)

    if args.wandb:
        import wandb

        wandb_args = args.__dict__.copy()
        wandb_args.update({f"config_{k}": v for k, v in game_config.items()})
        wandb.init(args.wandb_project_name, config=args)
        if args.name is not None:
            wandb.run.name = args.name
        else:
            args.name = wandb.run.name

    if args.name is None:
        args.name = "debug"

    return args, game_config
