import os
import random
from argparse import Namespace
from typing import Dict, List, Optional, Callable

import numpy as np
import pydicom
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from ldctbench.utils import load_yaml


class LDCTMayo(Dataset):
    """Dataset class for the LDCT dataset"""

    def __init__(self, mode: str, args: Namespace):
        """Init function

        Parameters
        ----------
        mode : str
            Which subset to use. Must be `train`, `val`, or `test`.
        args : Namespace
            Command line argumetns passed to the dataset class.
        """

        # Set seeds
        self.seed = args.seed
        np.random.seed(args.seed)
        random.seed(args.seed)

        self.path = args.datafolder
        if not hasattr(args, "eval_patchsize"):
            args.eval_patchsize = 128
        self.patchsize = (
            args.eval_patchsize if mode == "val" else args.patchsize
        )  # Always use same sized crops for validation independent of patchsize
        self.data_subset = args.data_subset
        self.data_norm = args.data_norm
        self.info = load_yaml(
            os.path.join(os.path.dirname(os.path.realpath(__file__)), "info.yml")
        )

        # Cache for sorted file paths to ensure alignment
        self.patient_cache = {}

        # Get all slices == samples for this split
        self.samples = [
            {**patient_dict, "slice_idx": s}
            for patient_dict in self.info[mode + "_set"]
            for s in range(patient_dict["n_slices"])
        ]
        random.shuffle(self.samples)

        self.weights = torch.tensor(
            [1.0 / patient_dict["n_slices"] for patient_dict in self.samples],
            dtype=torch.double,
        )

        if self.data_subset < 1.0:
            self.samples = self.samples[: int(len(self.samples) * self.data_subset)]
            self.weights = self.weights[: int(len(self.weights) * self.data_subset)]

    def _get_sorted_files(self, folder_rel_path):
        """Get list of DICOM files sorted by SliceLocation for correct alignment."""
        if folder_rel_path in self.patient_cache:
            return self.patient_cache[folder_rel_path]

        folder_abs_path = os.path.join(self.path, folder_rel_path[2:])
        if not os.path.exists(folder_abs_path):
            return []

        files = [f for f in os.listdir(folder_abs_path) if f.endswith(".dcm")]
        slice_info = []
        for f in files:
            p = os.path.join(folder_abs_path, f)
            try:
                ds = pydicom.dcmread(p, stop_before_pixels=True)
                loc = float(getattr(ds, "SliceLocation", 0))
                slice_info.append((loc, p))
            except Exception:
                slice_info.append((0, p))
        
        # Sort by SliceLocation
        slice_info.sort(key=lambda x: x[0])
        sorted_paths = [x[1] for x in slice_info]
        self.patient_cache[folder_rel_path] = sorted_paths
        return sorted_paths

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        if self.data_norm == "meanstd":
            return (X - self.info["mean"]) / self.info["std"]
        elif self.data_norm == "minmax":
            return (X - float(self.info["min"])) / (
                float(self.info["max"]) - float(self.info["min"])
            )
        else:
            raise ValueError(f"Unknown normalization method {self.data_norm}")

    def denormalize(self, X: np.ndarray) -> np.ndarray:
        if self.data_norm == "meanstd":
            return X * self.info["std"] + self.info["mean"]
        elif self.data_norm == "minmax":
            return X * (self.info["max"] - self.info["min"]) + self.info["min"]
        else:
            raise ValueError(f"Unknown normalization method {self.data_norm}")

    def _random_crop(self, images: List[np.ndarray]) -> List[np.ndarray]:
        assert images[0].shape == images[1].shape, "Both images must have same shape!"
        if (self.patchsize != images[0].shape[0] or self.patchsize != images[0].shape[1]) and self.patchsize:
            x = np.random.randint(images[0].shape[0] - self.patchsize)
            y = np.random.randint(images[0].shape[1] - self.patchsize)
            images = [im[x : x + self.patchsize, y : y + self.patchsize] for im in images]
        return images

    @staticmethod
    def to_torch(X: np.ndarray) -> torch.Tensor:
        return torch.unsqueeze(torch.from_numpy(X), 0)

    def reset_seed(self):
        np.random.seed(self.seed)
        random.seed(self.seed)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Get correctly aligned file paths
        input_files = self._get_sorted_files(sample["input"])
        target_files = self._get_sorted_files(sample["target"])
        
        # Ensure we don't go out of bounds if n_slices mismatch slightly
        slice_idx = sample["slice_idx"]
        if slice_idx >= len(input_files) or slice_idx >= len(target_files):
            slice_idx = 0 # Fallback
            
        x = pydicom.dcmread(input_files[slice_idx]).pixel_array.astype("float32")
        y = pydicom.dcmread(target_files[slice_idx]).pixel_array.astype("float32")

        x, y = self._random_crop([x, y])
        return {
            "x": self.to_torch(self._normalize(x)),
            "y": self.to_torch(self._normalize(y)),
        }


class TestData(Dataset):
    def __init__(self, datafolder, data_norm):
        self.info = load_yaml(
            os.path.join(os.path.dirname(os.path.realpath(__file__)), "info.yml")
        )
        self.data_norm = data_norm
        self.path = datafolder
        self.samples = []
        
        for patient_dict in tqdm(self.info["test_set"], desc="Load test patients"):
            input_folder = os.path.join(self.path, patient_dict["input"][2:])
            target_folder = os.path.join(self.path, patient_dict["target"][2:])
            
            # Sort slices by SliceLocation to ensure alignment
            input_files = self._get_sorted_paths(input_folder)
            target_files = self._get_sorted_paths(target_folder)
            
            n = min(len(input_files), len(target_files))
            
            patient_x, patient_y = [], []
            f_ld, f_hd = [], []
            
            for i in range(n):
                x = pydicom.dcmread(input_files[i]).pixel_array.astype("float32")
                y = pydicom.dcmread(target_files[i]).pixel_array.astype("float32")
                patient_x.append(x)
                patient_y.append(y)
                f_ld.append((patient_dict["input"][2:], os.path.basename(input_files[i])))
                f_hd.append((patient_dict["target"][2:], os.path.basename(target_files[i])))
                
            self.samples.append({
                "info": patient_dict,
                "x": np.stack(patient_x, axis=0),
                "y": np.stack(patient_y, axis=0),
                "f_ld": f_ld,
                "f_hd": f_hd
            })

    def _get_sorted_paths(self, folder_abs_path):
        if not os.path.exists(folder_abs_path): return []
        files = [f for f in os.listdir(folder_abs_path) if f.endswith(".dcm")]
        slice_info = []
        for f in files:
            p = os.path.join(folder_abs_path, f)
            try:
                ds = pydicom.dcmread(p, stop_before_pixels=True)
                loc = float(getattr(ds, "SliceLocation", 0))
                slice_info.append((loc, p))
            except:
                slice_info.append((0, p))
        slice_info.sort(key=lambda x: x[0])
        return [x[1] for x in slice_info]

    def _normalize(self, X):
        if self.data_norm == "meanstd":
            return (X - self.info["mean"]) / self.info["std"]
        elif self.data_norm == "minmax":
            return (X - float(self.info["min"])) / (
                float(self.info["max"]) - float(self.info["min"])
            )
        else:
            raise ValueError(f"Unknown normalization method {self.data_norm}")

    def denormalize(self, X):
        if self.data_norm == "meanstd":
            return X * self.info["std"] + self.info["mean"]
        elif self.data_norm == "minmax":
            return X * (self.info["max"] - self.info["min"]) + self.info["min"]
        else:
            raise ValueError(f"Unknown normalization method {self.data_norm}")

    def _convert_hu(self, X, to_hu=True):
        return X - 1024.0 if to_hu else X + 1024.0

    def to_torch(self, X):
        return torch.from_numpy(X)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "info": sample["info"],
            "x": self.to_torch(self._normalize(sample["x"])),
            "y": self.to_torch(self._normalize(sample["y"])),
            "f_hd": sample["f_hd"],
            "f_ld": sample["f_ld"],
        }
