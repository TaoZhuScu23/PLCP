import logging
import os
import random
import dgl
import numpy as np
import torch
import torch.nn.functional as F
import yaml


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


@torch.no_grad()
def validate(model, features, edge_index, labels, val_mask):
    model.eval()
    logits, _ = model(features, edge_index)
    loss_val = F.nll_loss(logits[val_mask], labels[val_mask]).item()
    acc_val = accuracy(logits[val_mask], labels[val_mask])
    return loss_val, acc_val


@torch.no_grad()
def test(model, features, edge_index, labels, test_mask):
    model.eval()
    logits, _ = model(features, edge_index)
    acc_test = accuracy(logits[test_mask], labels[test_mask])
    return acc_test, logits


def toDglGraph(edge_index):
    src_nodes = edge_index[0]
    dst_nodes = edge_index[1]

    g = dgl.graph((src_nodes, dst_nodes))
    g = dgl.add_self_loop(g)
    degs = g.in_degrees().float().clamp(min=1)
    norm = torch.pow(degs, -0.5).view(-1, 1)
    g.ndata['norm'] = norm
    g.apply_edges(lambda edges: {'m': edges.src['norm'] * edges.dst['norm']})
    return g


def pseudo_label_select(model, features, g, idx_unlabeled, args):
    model.eval()
    output = model(features, g)
    softmax_output = F.softmax(output, dim=-1)
    softmax_unlabeled = softmax_output[idx_unlabeled]

    _, pred_unlabeled = torch.max(softmax_unlabeled, 1)
    threshold_low, threshold_high = find_density_thresholds(softmax_output, idx_unlabeled, args.p1, args.p2)

    interval_indices = ((softmax_unlabeled.max(1)[0] >= threshold_low) & (
            softmax_unlabeled.max(1)[0] <= threshold_high)).nonzero(as_tuple=True)[0]
    interval_global_indices = idx_unlabeled[interval_indices]

    _, pseudo_labels = torch.max(F.softmax(output, dim=-1), 1)

    return interval_global_indices, pseudo_labels


def find_density_thresholds(softmaxed, idx, k1, k2):
    if k1 < 0:
        k1 = 0
    if k2 > 1:
        k2 = 1
    selected = softmaxed[idx]

    max_values, _ = torch.max(selected, dim=1)
    sorted_max_values, _ = torch.sort(max_values)

    total_samples = len(sorted_max_values)
    p1_idx = int(k1 * total_samples)
    p2_idx = int(k2 * total_samples)

    if k1 != 0:
        p1 = sorted_max_values[p1_idx].item()
    else:
        p1 = sorted_max_values[0].item()
    if k2 != 1:
        p2 = sorted_max_values[p2_idx - 1].item()
    else:
        p2 = sorted_max_values[-1].item()

    return p1, p2


def load_best_configs(args, path):
    with open(path, "r") as f:
        configs = yaml.load(f, yaml.FullLoader)

    if args.dataset not in configs:
        logging.info("Best args not found")
        return args

    logging.info("Using best configs")
    configs = configs[args.dataset]

    for k, v in configs.items():
        if "wd" in k:
            v = float(v)
        setattr(args, k, v)
    print("------ Use best configs ------")
    return args


def generate_seeds(n, start, end):
    if n > (end - start + 1):
        raise ValueError("Cannot generate a unique list with more elements than the specified range.")

    unique_numbers = set()
    while len(unique_numbers) < n:
        unique_numbers.add(random.randint(start, end))

    return list(unique_numbers)

def consis_loss(args, logps, tem, conf):
    ps = [torch.exp(p) for p in logps]
    sum_p = 0.
    for p in ps:
        sum_p = sum_p + p
    avg_p = sum_p / len(ps)

    sharp_p = (torch.pow(avg_p, 1. / tem) / torch.sum(torch.pow(avg_p, 1. / tem), dim=1, keepdim=True)).detach()
    loss = 0.
    for p in ps:
        if args.loss == 'kl':
            loss += torch.mean((-sharp_p * torch.log(p)).sum(1)[avg_p.max(1)[0] > conf])
        elif args.loss == 'l2':
            loss += torch.mean((p - sharp_p).pow(2).sum(1)[avg_p.max(1)[0] > conf])
        else:
            raise ValueError(f"Unknown loss type: {args.loss}")
    loss = loss / len(ps)
    return loss


def clip_grad_norm(params, max_norm):
    if max_norm > 0:
        return torch.nn.utils.clip_grad_norm_(params, max_norm)
    else:
        return torch.sqrt(sum(p.grad.data.norm() ** 2 for p in params if p.grad is not None))
