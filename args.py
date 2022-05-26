from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser(description="Matman QA")
    parser.add_argument('--cuda', action='store_false', help='do not use cuda', dest='cuda', default=False)
    parser.add_argument('--gpu', type=int, default=-1) # Use -1 for CPU
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--mode', type=str, default='static')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=3435)
    parser.add_argument('--dataset', type=str, default='trec')
    parser.add_argument('--resume_snapshot', type=str, default=None)
    parser.add_argument('--dev_every', type=int, default=5)
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--save_path', type=str,    default='saves')    
    parser.add_argument('--epoch_decay', type=int, default=15)    
    parser.add_argument('--weight_decay',type=float, default=1e-5)
    parser.add_argument('--neg_num', type=int, default=8)
    parser.add_argument('--neg_sample', type=str, default="random")
    parser.add_argument('--eps', type=float, default=1e-6)
    parser.add_argument('--optimizer', type=str, default="sgd")
    parser.add_argument('--tensorboard', action='store_true', default=False,
                        help='use TensorBoard to visualize training (default: false)')
    parser.add_argument('--emb_dim', type=int, default=14, help='dimension of embedding SPD matrices')
    parser.add_argument('--n_dim', type=int, default=14, help='dimension n of projection matrices')
    parser.add_argument('--p_dim', type=int, default=7, help='dimension p of projection matrices')
    parser.add_argument('--dist_factor', type=float, default=2.0, help='Weighted factor of spd and gr parts')
    parser.add_argument('--metric', type=str, default="riem", help='riem,fone')
    parser.add_argument("--regularizer", choices=["N3", "F2"], default="F2", help="Regularizer")
    parser.add_argument("--regularizer_weight", default=0.2, type=float, help="Regularization weight")

    args = parser.parse_args()
    return args
