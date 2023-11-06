import argparse


def arg_parse():
    parser = argparse.ArgumentParser(description='Experiment with DomainNet')
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--batch_size', '-b', type=int, required=True)
    parser.add_argument('--epoch', '-e', type=int, required=True)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--save_dir', type=str, default='result',
                        help='save directory for result and loss')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--device', default=0, type=int)


    return parser.parse_args()
