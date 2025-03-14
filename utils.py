import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import json


def generate_plots(list_of_dirs, legend_names, save_path):
    """ Generate plots according to log 
    :param list_of_dirs: List of paths to log directories
    :param legend_names: List of legend names
    :param save_path: Path to save the figs
    """
    assert len(list_of_dirs) == len(legend_names), "Names and log directories must have same length"
    data = {}
    for logdir, name in zip(list_of_dirs, legend_names):
        json_path = os.path.join(logdir, 'results.json')
        assert os.path.exists(os.path.join(logdir, 'results.json')), f"No json file in {logdir}"
        with open(json_path, 'r') as f:
            data[name] = json.load(f)
    
    for yaxis in ['train_accs', 'valid_accs', 'train_losses', 'valid_losses']:
        fig, ax = plt.subplots()
        for name in data:
            ax.plot(data[name][yaxis], label=name)
        ax.legend()
        ax.set_xlabel('epochs')
        ax.set_ylabel(yaxis.replace('_', ' '))
        fig.savefig(os.path.join(save_path, f'{yaxis}.png'))
        

def seed_experiment(seed):
    """Seed the pseudorandom number generator, for repeatability.

    Args:
        seed (int): random seed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True


def to_device(tensors, device):
    if isinstance(tensors, torch.Tensor):
        return tensors.to(device=device)
    elif isinstance(tensors, dict):
        return dict(
            (key, to_device(tensor, device)) for (key, tensor) in tensors.items()
        )
    elif isinstance(tensors, list):
        return list(
            (to_device(tensors[0], device), to_device(tensors[1], device)))
    else:
        raise NotImplementedError("Unknown type {0}".format(type(tensors)))


# def cross_entropy_loss(logits: torch.Tensor, labels: torch.Tensor):
#     """ Return the mean loss for this batch
#     :param logits: [batch_size, num_class]
#     :param labels: [batch_size]
#     :return loss 
#     """
#     raise NotImplementedError
def cross_entropy_loss(logits: torch.Tensor, labels: torch.Tensor):
    """ Compute cross-entropy loss manually without torch.nn.CrossEntropyLoss
    :param logits: [batch_size, num_classes] (non normalisés, output du modèle)
    :param labels: [batch_size] (indices des classes correctes)
    :return: Moyenne de la cross-entropy loss sur le batch
    """
    # Étape 1: Appliquer softmax sur les logits pour obtenir des probabilités
    probs = torch.nn.functional.softmax(logits, dim=1)  # [batch_size, num_classes]
    
    # Étape 2: Récupérer les probabilités associées aux labels réels
    batch_size = logits.shape[0]
    true_probs = probs[range(batch_size), labels]  # [batch_size]
    
    # Étape 3: Calcul de la log-likelihood
    log_likelihood = torch.log(true_probs + 1e-9)  # Éviter log(0) avec un petit epsilon
    
    # Étape 4: Calcul de la cross-entropy loss (moyenne sur le batch)
    loss = -log_likelihood.mean()
    
    return loss

def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor):
    """ Compute the accuracy of the batch """
    acc = (logits.argmax(dim=1) == labels).float().mean()
    return acc
