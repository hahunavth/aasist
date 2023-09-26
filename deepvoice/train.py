# import sys
# # add root dir to path
# root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# if root_dir not in sys.path:
#     print(f"Add {root_dir} to sys.path")
#     sys.path.append(root_dir)
# # add current dir to path
# if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
#     print(f"Add {os.path.dirname(os.path.abspath(__file__))} to sys.path")
#     sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import random, os
from pprint import pprint
from types import SimpleNamespace
from typing import Dict, List, Union
import soundfile as sf

import numpy as np
import torch
from torch import Tensor
from hyperpyyaml import load_hyperpyyaml
from torch.utils.data import DataLoader, Dataset
from torchcontrib.optim import SWA

from utils import create_optimizer


def set_seed(seed, cudnn_deterministic=True, cudnn_benchmark=False):
    """
    set initial seed for reproduction
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = cudnn_benchmark


def get_model_tag(
        prefix: str = None,
        batch_size: int = None,
        epochs: int = None,
        postfix: str = None
):
    """
    Generate a tag for the model based on the hyperparameters
    """
    assert all([prefix, batch_size, epochs]), "prefix, batch_size, epochs must be provided"

    model_tag = f"ep{epochs}_bs{batch_size}"
    if prefix:
        model_tag = f"{prefix}_{model_tag}"
    if postfix:
        model_tag = f"{model_tag}_{postfix}"
    return model_tag


def genSpoof_list(audio_dir, real_dir="REAL", fake_dir="FAKE"):
    fake = os.listdir(os.path.join(audio_dir, fake_dir))
    real = os.listdir(os.path.join(audio_dir, real_dir))

    d_meta = {fname: 1 for fname in real}
    d_meta.update({fname: 0 for fname in fake})
    file_list = [*fake, *real]

    return d_meta, file_list


def pad_random(x: np.ndarray, max_len: int = 64600):
    x_len = x.shape[0]
    # if duration is already long enough
    if x_len >= max_len:
        stt = np.random.randint(x_len - max_len)
        return x[stt:stt + max_len]

    # if too short
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, num_repeats)[:max_len]
    return padded_x


class DeepVoiceDataset(Dataset):
    def __init__(self, list_fname, labels, base_dir, real_dir='REAL', fake_dir='FAKE'):
        self.list_fname = list_fname
        self.labels = labels
        self.base_dir = base_dir
        self.real_dir = real_dir
        self.fake_dir = fake_dir
        self.cut = 64600  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_fname)

    def __getitem__(self, index):
        fname = self.list_fname[index]
        y = self.labels[fname]
        X, _ = sf.read(os.path.join(self.base_dir, self.real_dir if y == 1 else self.fake_dir, fname))
        X_pad = pad_random(X, self.cut)
        x_inp = Tensor(X_pad)

        return x_inp, y

    @staticmethod
    def from_dir(lb_cont_dir):
        """
        Load dataset from directory
        :param lb_cont_dir: directory containing the label with name is the label and content is the list of file name
        """
        d_meta, file_list = genSpoof_list(lb_cont_dir)
        return DeepVoiceDataset(file_list, d_meta, lb_cont_dir)


def get_loader(
        train_dir: str,
        valid_dir: str,
        test_dir: str,
        seed: int,
        batch_size: int) -> List[torch.utils.data.DataLoader]:

    gen = torch.Generator()
    gen.manual_seed(seed)

    train_set = DeepVoiceDataset.from_dir(train_dir)
    dev_set = DeepVoiceDataset.from_dir(valid_dir)
    test_set = DeepVoiceDataset.from_dir(test_dir)

    def seed_worker(worker_id):
        """
        Used in generating seed for the worker of torch.utils.data.Dataloader
        """
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    trn_loader = DataLoader(train_set,
                            batch_size=batch_size,
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True,
                            worker_init_fn=seed_worker,
                            generator=gen)
    dev_loader = DataLoader(dev_set,
                            batch_size=batch_size,
                            shuffle=False,
                            drop_last=False,
                            pin_memory=True)
    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             drop_last=False,
                             pin_memory=True)

    return trn_loader, dev_loader, test_loader


def produce_evaluation_file(
        data_loader: DataLoader,
        model,
        device: torch.device,
        save_path: str) -> None:
    """Perform evaluation and save the score to a file"""
    model.eval()
    fname_list = []
    score_list = []
    lb_list = []
    for batch_x, batch_y in data_loader:
        batch_x = batch_x.to(device)
        with torch.no_grad():
            _, batch_out = model(batch_x)
            batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
        # add outputs
        #         fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())
        lb_list.extend(batch_y)

    with open(save_path, "w") as fh:
        # todo: fname, fname_list,
        for score, lb in zip(score_list, lb_list):
            fh.write("{} {}\n".format(score, lb))
    print("Scores saved to {}".format(save_path))


def calculate_EER(cm_scores_file="/kaggle/working/name_ep1_bs24/eval_scores_using_best_dev_model.txt"):
    cm_data = np.genfromtxt(cm_scores_file, dtype=str)
    cm_keys = cm_data[:, 1]
    cm_scores = cm_data[:, 0].astype(np.float64)

    bona_cm = cm_scores[cm_keys == '1'] # 'bonafide'
    spoof_cm = cm_scores[cm_keys == '0'] # 'spoof'

    from evaluation import compute_eer
    eer_cm = compute_eer(bona_cm, spoof_cm)[0]

    return eer_cm * 100


def train(
        model: torch.nn.Module,
        trn_loader: DataLoader,
        dev_loader: DataLoader,
        test_loader: DataLoader,
        optimizer,
        scheduler,
        optimizer_swa,
        writer,
        model_tag_dir,
        hparams
):
    # train
    best_dev_eer = 1000.  # 1.
    best_eval_eer = 100.
    best_dev_tdcf = 0.05
    best_eval_tdcf = 1.
    n_swa_update = 0  # number of snapshots of model to use in SWA
    f_log = open(os.path.join(model_tag_dir, "metric_log.txt"), "a")
    f_log.write("=" * 5 + "\n")

    metric_dir = hparams["metric_dir"]
    os.makedirs(metric_dir, exist_ok=True)

    # todo: refactor in train_epoch fn
    if "eval_all_best" not in hparams:
        hparams["eval_all_best"] = "True"
    if "freq_aug" not in hparams:
        hparams["freq_aug"] = "False"

    for epoch in range(hparams["epochs"]):
        print("Start training epoch{:03d}".format(epoch))

        # todo
        from main import train_epoch
        running_loss = train_epoch(trn_loader, model, optimizer, device, scheduler, hparams)
        produce_evaluation_file(dev_loader, model, device, eval_score_file)

        # todo: calculate_tDCF_EER
        dev_eer = calculate_EER(eval_score_file)
        print("DONE.")
        print("Loss:{:.5f}, dev_eer: {:.3f}".format(running_loss, dev_eer))

        writer.add_scalar("loss", running_loss, epoch)
        writer.add_scalar("dev_eer", dev_eer, epoch)

        if best_dev_eer >= dev_eer:
            print("best model find at epoch", epoch)
            best_dev_eer = dev_eer
            torch.save(
                model.state_dict(),
                os.path.join(ckpt_dir, "epoch_{}_{:03.3f}.pth").format(epoch, dev_eer)
            )

            # todo: do evaluation whenever best model is renewed
            #

            print("Saving epoch {} for swa".format(epoch))
            optimizer_swa.update_swa()
            n_swa_update += 1
        writer.add_scalar("best_dev_eer", best_dev_eer, epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--hparams",
                        dest="hparams",
                        type=str,
                        help="hparams file",
                        default="./train.yaml",
                        # required=True
                        )
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        type=str,
        help="output directory for results",
        default="./exp_result",
    )
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="random seed (default: 1234)")
    parser.add_argument(
        "--eval",
        action="store_true",
        help="when this flag is given, evaluates given model and exit")
    parser.add_argument("--comment",
                        type=str,
                        default=None,
                        help="comment to describe the saved model")
    # parser.add_argument("--eval_model_weights",
    #                     type=str,
    #                     default=None,
    #                     help="directory to the model weight file (can be also given in the config file)")
    args = parser.parse_args()

    with open(args.hparams, "r") as f:
        hparams = load_hyperpyyaml(f, {
            "output_dir": args.output_dir,
            "seed": args.seed,
            "comment": args.comment,
            "eval": args.eval
        })
        pprint(hparams)

        # torch device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device: {}".format(device))
        # if device != "cuda":
        #     raise Exception("Cuda is not available!")

        # tag
        model_tag = hparams["model_tag"]
        # dir
        output_dir = hparams["output_dir"]
        model_tag_dir = hparams["model_tag_dir"]
        ckpt_dir = hparams["checkpoint_dir"]
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(model_tag_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)

        eval_score_file = hparams['eval_score_file']

        writer = hparams["writer"]

        # get model
        model = hparams['model']
        model = model.to(device)

        # dataloader
        trn_loader, dev_loader, test_loader = hparams["dataloader_list"]

        # get optimizer and scheduler
        optim_config = hparams["optim_config"]
        optim_config["steps_per_epoch"] = len(trn_loader)
        optimizer, scheduler = create_optimizer(model.parameters(), optim_config)
        optimizer_swa = SWA(optimizer)

        # train
        if not hparams["eval"]:
            train(
                model,
                trn_loader,
                dev_loader,
                test_loader,
                optimizer,
                scheduler,
                optimizer_swa,
                writer,
                model_tag_dir,
                hparams
            )
        else:
            # todo: evaluate
            pass
