import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from pathlib import Path

from inosgd.utils.saveload import load


class _IndexedSurnames(Dataset):
    def __init__(self, train=True, indexed=True):
        train_set, test_set = load("Datasets/surnames.pkl")

        if not train:
            indices = test_set.indices
            self.data = [test_set.dataset.samples[i][0] for i in indices]
            self.targets = torch.Tensor([test_set.dataset.samples[i][1] for i in indices]).int()
        else:
            indices = train_set.indices
            self.data = [train_set.dataset.samples[i][0] for i in indices]
            if not indexed:
                self.targets = torch.Tensor([train_set.dataset.samples[i][1] for i in indices]).int()
            else:
                self.targets = torch.Tensor([[train_set.dataset.samples[i][1], j] for j, i in enumerate(indices)]).int()
        

    def __len__(self):
        return len(self.targets)
    

    def __getitem__(self, index):
        datum, target = self.data[index], self.targets[index].int()
        return datum, target


def load_surnames(which: str):
    if which == 'train':
        return _IndexedSurnames(train=True, indexed=True)
    elif which == 'test':
        return _IndexedSurnames(train=False, indexed=False)
    else:
        raise ValueError("Can only choose between 'train' and 'test'.")
    

##### https://opacus.ai/tutorials/building_lstm_name_classifier
class CharByteEncoder(nn.Module):
    """
    This encoder takes a UTF-8 string and encodes its bytes into a Tensor. It can also
    perform the opposite operation to check a result.
    Examples:
    >>> encoder = CharByteEncoder()
    >>> t = encoder('Ślusàrski')  # returns tensor([256, 197, 154, 108, 117, 115, 195, 160, 114, 115, 107, 105, 257])
    >>> encoder.decode(t)  # returns "<s>Ślusàrski</s>"
    """

    def __init__(self):
        super().__init__()
        self.start_token = "<s>"
        self.end_token = "</s>"
        self.pad_token = "<pad>"

        self.start_idx = 256
        self.end_idx = 257
        self.pad_idx = 258

    def forward(self, s: str, pad_to=0) -> torch.LongTensor:
        """
        Encodes a string. It will append a start token <s> (id=self.start_idx) and an end token </s>
        (id=self.end_idx).
        Args:
            s: The string to encode.
            pad_to: If not zero, pad by appending self.pad_idx until string is of length `pad_to`.
                Defaults to 0.
        Returns:
            The encoded LongTensor of indices.
        """
        encoded = s.encode()
        n_pad = pad_to - len(encoded) if pad_to > len(encoded) else 0
        return torch.LongTensor(
            [self.start_idx]
            + [c for c in encoded]  # noqa
            + [self.end_idx]
            + [self.pad_idx for _ in range(n_pad)]
        )

    def decode(self, char_ids_tensor: torch.LongTensor) -> str:
        """
        The inverse of `forward`. Keeps the start, end, and pad indices.
        """
        char_ids = char_ids_tensor.cpu().detach().tolist()

        out = []
        buf = []
        for c in char_ids:
            if c < 256:
                buf.append(c)
            else:
                if buf:
                    out.append(bytes(buf).decode())
                    buf = []
                if c == self.start_idx:
                    out.append(self.start_token)
                elif c == self.end_idx:
                    out.append(self.end_token)
                elif c == self.pad_idx:
                    out.append(self.pad_token)

        if buf:  # in case some are left
            out.append(bytes(buf).decode())
        return "".join(out)

    def __len__(self):
        """
        The length of our encoder space. This is fixed to 256 (one byte) + 3 special chars
        (start, end, pad).
        Returns:
            259
        """
        return 259


class NamesDataset(Dataset):
    def __init__(self, root):
        self.root = Path(root)

        self.labels = list({langfile.stem for langfile in self.root.iterdir()})
        self.labels_dict = {label: i for i, label in enumerate(self.labels)}
        self.encoder = CharByteEncoder()
        self.samples = self.construct_samples()

    def __getitem__(self, i):
        return self.samples[i]

    def __len__(self):
        return len(self.samples)

    def construct_samples(self):
        samples = []
        for langfile in self.root.iterdir():
            label_name = langfile.stem
            label_id = self.labels_dict[label_name]
            with open(langfile, "r") as fin:
                for row in fin:
                    samples.append(
                        (self.encoder(row.strip()), torch.tensor(label_id).long())
                    )
        return samples

    def label_count(self):
        cnt = Counter()
        for _x, y in self.samples:
            label = self.labels[int(y)]
            cnt[label] += 1
        return cnt
        