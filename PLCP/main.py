import copy
import datetime
import argparse
from model import GRAND
import math
from tqdm import tqdm
from dataloader import load_dataset
from utils import *


def train(model, g, features, optimizer, train_mask, labels, num_classes, idx_val, idx_unlabeled, trial,
          selected_pseudo_indices=None,
          pseudo_labels=None, retrain=False):
    best_val_loss = 9999999
    bad_counter = 0
    best_loss_val_acc = 0
    best_model = model
    s = "Pre-training" if retrain is False else "Training"
    pbar = tqdm(range(1, 2500), desc=f'{s} on runs {trial}',
                bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', ncols=180)
    for epoch in pbar:
        X = features
        model.train()
        optimizer.zero_grad()
        K = args.sample
        loss_train = 0.
        output_list = []
        for k in range(K):
            output_list.append(torch.log_softmax(model(X, g), dim=-1))
            if retrain is False:
                loss_train += F.nll_loss(output_list[k][train_mask], labels[train_mask])
            else:
                loss_train += F.nll_loss(output_list[k][train_mask],
                                         labels[train_mask]) + args.lam_loss * F.nll_loss(
                    output_list[k][selected_pseudo_indices], pseudo_labels[selected_pseudo_indices])
        loss_train = loss_train / K
        loss_consis = min(args.lam, (args.lam * float(epoch) / args.warmup)) * consis_loss(args, output_list,
                                                                                           args.tem,
                                                                                           2. / num_classes)
        loss_train = loss_train + loss_consis
        acc_train = accuracy(output_list[0][train_mask], labels[train_mask])
        loss_train.backward()
        clip_grad_norm(model.parameters(), args.clip_norm)
        optimizer.step()
        model.eval()
        output = torch.log_softmax(model(X, g), dim=-1)
        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])
        loss_train = loss_train.item()

        pbar.set_postfix(loss=f'{loss_train:.4f}', best_val_loss=f'{best_val_loss:.4f}',
                         best_acc_val=f'{best_loss_val_acc:.4f}', train_acc=f'{acc_train:.4f}')
        if loss_val < best_val_loss:
            best_val_loss = loss_val
            best_loss_val_acc = acc_val
            bad_counter = 0
            best_model = copy.deepcopy(model)
            if retrain is True and epoch >= args.pseudo_warmup:
                selected_pseudo_indices, pseudo_labels = pseudo_label_select(
                    model, features, g,
                    idx_unlabeled, args)
        else:
            bad_counter += 1
        if bad_counter == args.patience:
            break
    return best_model


def main(args):
    setup_seed(0)
    data = load_dataset(args.dataset).cuda()
    features, labels, edge_index, train_mask, val_mask, test_mask = data.x, data.y, data.edge_index, data.train_mask, data.val_mask, data.test_mask
    num_classes = torch.max(labels).item() + 1
    unlabeled_mask = (~train_mask) & (~val_mask)
    idx_unlabeled = torch.nonzero(unlabeled_mask).squeeze(1)
    idx_val = torch.nonzero(val_mask).squeeze(1)

    feature, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
    final_acc_list = []

    if args.model_saving is True:
        output_dir = "trained_models"
        os.makedirs(output_dir, exist_ok=True)
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        sub_dir = os.path.join(output_dir, current_time)
        os.makedirs(sub_dir, exist_ok=True)
    g = toDglGraph(edge_index)
    for trial in range(len(args.seeds)):
        setup_seed(args.seeds[trial])
        model = GRAND(features.shape[1], num_classes, args.hidden, args.input_droprate, args.hidden_droprate,
                      args.dropnode_rate, args.order,
                      args.layer_num,
                      args.use_bn, args.node_norm).cuda()
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.lr, weight_decay=args.wd)
        best_model = train(model, g, features, optimizer, train_mask, labels, num_classes, idx_val, idx_unlabeled,
                           trial)

        best_model.eval()
        selected_pseudo_indices, pseudo_labels = pseudo_label_select(
            model, features, g,
            idx_unlabeled, args)
        model = best_model
        model.dropnode_rate = args.after_dropnode_rate
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.after_lr, weight_decay=args.after_wd)
        best_model = train(model, g, features, optimizer, train_mask, labels, num_classes, idx_val, idx_unlabeled,
                           trial,
                           selected_pseudo_indices,
                           pseudo_labels, True)

        best_model.eval()
        output = torch.log_softmax(best_model(features, g), dim=-1)
        final_acc = accuracy(output[test_mask], labels[test_mask])
        final_acc_list.append(round(final_acc.cpu().numpy() * 100, 2))
        print("Test Acc: ", round(final_acc.cpu().numpy() * 100, 2))
        print("-----------------------------------------------------------------------------")
        if args.model_saving is True:
            model_path = os.path.join(sub_dir,
                                      f"model_seed_{args.seeds[trial]}_{args.dataset}_{math.floor(final_acc.cpu().numpy() * 10000)}.pth")
            torch.save(best_model.state_dict(), model_path)

    if args.model_saving is True:
        new_sub_dir_name = f"{current_time}_{math.floor(round(np.mean(final_acc_list), 2) * 100)}_{len(args.seeds)}seeds"
        new_sub_dir_path = os.path.join(output_dir, new_sub_dir_name)
        if not os.path.exists(new_sub_dir_path):
            os.rename(sub_dir, new_sub_dir_path)
    print(f'Final Acc: {round(np.mean(final_acc_list), 2)} ± {round(np.std(final_acc_list), 2)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int,
                        default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                        help='Random seed.')
    parser.add_argument('--dev', type=int, default=0, help='Device id')
    parser.add_argument('--epochs', type=int, default=2500, help='Number of epochs to train.')
    parser.add_argument('--patience', type=int, default=200, help='Patience')

    parser.add_argument('--dataset', default='Cora', help='Dateset')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--wd', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--layer_num', type=int, default=2, help='MLP layer num')
    parser.add_argument('--hidden', type=int, default=256, help='Hidden dimensions.')
    parser.add_argument('--samples', type=int, default=4, help="Augmentation times")
    parser.add_argument('--tem', type=float, default=0.7, help='Sharpening temperature')
    parser.add_argument('--lam', type=float, default=0.3, help='Lamda')
    parser.add_argument('--dropnode_rate', type=float, default=0.5,
                        help='Dropnode rate (1 - keep probability).')
    parser.add_argument('--input_droprate', type=float, default=0.,
                        help='Dropout rate of the input layer (1 - keep probability).')
    parser.add_argument('--hidden_droprate', type=float, default=0.2,
                        help='Dropout rate of the hidden layer (1 - keep probability).')
    parser.add_argument('--sample', type=int, default=2, help='Sampling times of dropnode')
    parser.add_argument('--use_bn', action='store_true', default=False, help='Using Batch Normalization')
    parser.add_argument('--node_norm', action='store_true', default=False, help='Embedding L2 normalization')
    parser.add_argument("--clip_norm", type=float,
                        default=-1, help="clip norm")
    parser.add_argument('--order', type=int, default=2, help='Propagation step N')
    parser.add_argument('--loss', type=str, default='l2', help="Consistency loss function, l2 or kl")
    parser.add_argument('--warmup', type=int, default=400, help='Consistency loss warmup')

    parser.add_argument('--lam_loss', type=float, default=2.9, help="Loss weight")
    parser.add_argument('--p1', type=float, default=0.2, help="Pseudo-label selection range [p1, p2], p2 for high")
    parser.add_argument('--p2', type=float, default=0.3, help="Pseudo-label selection range [p1, p2], p1 for low")
    parser.add_argument('--pseudo_warmup', type=int, default=10, help="Pseudo-label selection warmup")
    parser.add_argument('--after_dropnode_rate', type=float, default=0.5,
                        help='Dropnode rate (1 - keep probability) after pre-training')
    parser.add_argument('--after_lr', type=float, default=0.01, help='Learning rate after pre-training')
    parser.add_argument('--after_wd', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters) after pre-training')

    parser.add_argument('--model_saving', type=bool, default=False)

    args = parser.parse_args()
    args = load_best_configs(args, "configs.yml")
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device("cuda:1")
    if args.p1 >= args.p2:
        raise ValueError(f"Incorrect {args.k1} and {args.k2}!")

    # args.seeds = generate_seeds(10, 0, 10000000)

    main(args)
