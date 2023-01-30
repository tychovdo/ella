import numpy as np
import torch
from torch.utils.data import Dataset
import pickle
from sklearn.datasets import make_moons
from typing import Union, Callable 
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets

LOG2 = np.log(2.0)


def get_circle_data(n_obs=36, degree=90, sigma_noise=0.05, seed=1):
    """ Generate dummy circle data.

    Args:
        n_obs: Total number of observations
        degree: Angle in degrees
        sigma_noise: Amount of observation noise
        seed: Integer used as random seed

    """

    torch.manual_seed(seed)

    rads = np.deg2rad(degree)

    # Divide total number of observations evenly over three rings
    n_obs_x0 = int(n_obs // 2)
    n_obs_x1 = n_obs - n_obs_x0
    n_obs_x1_outer = int(n_obs_x1 // (3 / 2))
    n_obs_x1_inner = n_obs_x1 - n_obs_x1_outer

    # Generate data points and transform into rings
    X0_t = torch.rand(n_obs_x0) * rads
    X0 = torch.stack([torch.cos(X0_t), torch.sin(X0_t)]).T * (0.5 + torch.randn(n_obs_x0, 2) * sigma_noise)

    X1_inner = torch.randn(n_obs_x1_inner, 2) * sigma_noise
    X1_outer_t = torch.rand(n_obs_x1_outer) * rads
    X1_outer = torch.stack([torch.cos(X1_outer_t), torch.sin(X1_outer_t)]).T * (1.0 + torch.randn(n_obs_x1_outer, 2) * sigma_noise)
    X1 = torch.cat((X1_inner, X1_outer))

    # Generate labels
    X = torch.cat((X0, X1))
    y = (torch.arange(0, n_obs) >= n_obs_x0).long()

    return X, y, None


class SnelsonGen(Dataset):
    def __init__(
        self,
        root='',
        train=True,
        n_samples=150,
        s_noise=0.3,
        random_seed=6,
        filter_left=True,
        filter_right=False,
        double=False,
    ):
        self.train = train
        self.seed = random_seed if train else random_seed - 1
        self.n_samples = n_samples
        self.s_noise = s_noise
        self.filter_left = filter_left
        self.filter_right = filter_right
        with open(root + '/snelson/fsnelson.spy', 'rb') as file:
            self.fsnelson = pickle.load(file)
        X, y = self.generate()
        if double:
            self.data = torch.from_numpy(X).double()
            self.targets = torch.from_numpy(y).double()
        else:
            self.data = torch.from_numpy(X).float()
            self.targets = torch.from_numpy(y).float()

    def generate(self):
        samples = self.n_samples * 10
        np.random.seed(self.seed)
        x_min, x_max = self.x_bounds
        xs = (x_max - x_min) * np.random.rand(samples) + x_min
        fs = self.fsnelson(xs)
        ys = fs + np.random.randn(samples) * self.s_noise
        if self.filter_left:
            xfilter = (xs <= 1.5) | (xs >= 2.4)
            xs, ys = xs[xfilter], ys[xfilter]
        if self.filter_right:
            xfilter = (xs <= 4.4) | (xs >= 5.2)
            xs, ys = xs[xfilter], ys[xfilter]
        xs, ys = xs[: self.n_samples], ys[: self.n_samples]
        return xs.reshape(-1, 1), ys

    @property
    def x_bounds(self):
        x_min, x_max = (0.059167804, 5.9657729)
        return x_min, x_max

    @property
    def y_bounds(self):
        return min(self.targets).item(), max(self.targets).item()

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return self.n_samples

    def grid(self, density=250, offset=1.5):
        x_min, x_max = self.x_bounds
        return torch.linspace(x_min-offset, x_max+offset, steps=density).unsqueeze(-1)


class Snelson(Dataset):
    def __init__(self, root='', train=True, double=False):
        self.train = train

        if self.train:
            train_inputs = self.load_snelson('train_inputs', root)
            train_outputs = self.load_snelson('train_outputs', root)
            self.targets = torch.from_numpy(train_outputs)
            self.data = torch.from_numpy(train_inputs.reshape(-1, 1))
        else:
            test_inputs = self.load_snelson('test_inputs', root)
            self.data = torch.from_numpy(test_inputs.reshape(-1, 1))
            self.targets = torch.from_numpy(np.zeros_like(test_inputs))

        if not double:
            self.data = self.data.float()
            self.targets = self.targets.float()

    @staticmethod
    def load_snelson(file_name, root):
        with open(root + f'/snelson/{file_name}', 'r') as f:
            arr = [float(i) for i in f.read().strip().split('\n')]
            return np.array(arr)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return self.data.shape[0]

    @property
    def x_bounds(self):
        return min(self.data).item(), max(self.data).item()

    @property
    def y_bounds(self):
        return min(self.targets).item(), max(self.targets).item()

    def grid(self, density=250, offset=1.5):
        x_min, x_max = self.x_bounds
        return torch.linspace(x_min-offset, x_max+offset, steps=density).unsqueeze(-1)


class TwoMoons(Dataset):
    def __init__(
        self, train=True, random_seed=6, noise=0.3, n_samples=150, double=False
    ):
        self.train = train
        self.seed = random_seed if train else random_seed - 1
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=self.seed)
        self.C = 2  # binary problem

        if double:
            self.data = torch.from_numpy(X).double()
            self.targets = torch.from_numpy(y).double()
        else:
            self.data = torch.from_numpy(X).float()
            self.targets = torch.from_numpy(y).float()

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return self.data.shape[0]

    @property
    def bounding_box(self):
        x1_min, x1_max = self.data[:, 0].min(), self.data[:, 0].max()
        x2_min, x2_max = self.data[:, 1].min(), self.data[:, 1].max()
        return ((x1_min, x1_max), (x2_min, x2_max))

    def grid(self, density=100, offset=1.0):
        ((x1_min, x1_max), (x2_min, x2_max)) = self.bounding_box
        xx, yy = np.meshgrid(
            np.linspace(x1_min-offset, x1_max+offset, density),
            np.linspace(x2_min-offset, x2_max+offset, density),
        )
        grid = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()])
        return grid, xx, yy


class RotatedMNIST(datasets.MNIST):
    """ Rotated MNIST class.
        Wraps regular pytorch MNIST and rotates each image randomly between given angle using fixed seed. """

    def __init__(self, root: str, degree: float, mode: str = 'bilinear',
                 train: bool = True, transform: Union[Callable, type(None)] = None,
                 target_transform: Union[Callable, type(None)] = None, download: bool = False):
        super().__init__(root, train, transform, target_transform, download)
        """
        Args:
            root (string): Root directory of dataset where ``MNIST/processed/training.pt``
                and  ``MNIST/processed/test.pt`` exist.
            degree (float): Amount of rotation in degrees
            mode (float): Mode used for interpolation (nearest|bilinear|bicubic)
            train (bool, optional): If True, creates dataset from ``training.pt``,
                otherwise from ``test.pt``.
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
                same arguments as torchvision.see arguments of torchvision.dataset.MNIST)
        """
        # Use fixed seed (0 for test set, 1 for training set)
        torch.manual_seed(int(train))

        # Sample radians
        rad = np.radians(degree)
        thetas = (torch.rand(len(self.data)) * 2 - 1) * rad

        # Create rotation matrices
        c, s = torch.cos(thetas), torch.sin(thetas)
        rot_matrices = torch.stack((torch.stack((c, -s, c*0), 0),
                                    torch.stack((s, c, s*0), 0)), 0).permute(2, 0, 1)

        # Create resampling grid
        rot_grids = F.affine_grid(rot_matrices, self.data.unsqueeze(1).shape, align_corners=False)

        # Rotate images using 'mode' interpolation (e.g. bilinear sampling)
        xmin, xmax = self.data.min(), self.data.max()
        data_dtype = self.data.dtype

        self.data = F.grid_sample(self.data.unsqueeze(1).float(), rot_grids, align_corners=False, mode=mode)
        self.data = self.data[:, 0].type(data_dtype).clamp(xmin, xmax)


class RotatedFashionMNIST(datasets.FashionMNIST):
    """ Rotated Fashion MNIST dataset.
        Wraps regular pytorch FashionMNIST and rotates each image randomly between given angle using fixed seed. """

    def __init__(self, root: str, degree: float, mode: str = 'bilinear',
                 train: bool = True, transform: Union[Callable, type(None)] = None,
                 target_transform: Union[Callable, type(None)] = None, download: bool = False):
        super().__init__(root, train, transform, target_transform, download)
        """
        Args:
            root (string): Root directory of dataset where ``MNIST/processed/training.pt``
                and  ``MNIST/processed/test.pt`` exist.
            degree (float): Amount of rotation in degrees
            mode (float): Mode used for interpolation (nearest|bilinear|bicubic)
            train (bool, optional): If True, creates dataset from ``training.pt``,
                otherwise from ``test.pt``.
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
                same arguments as torchvision.see arguments of torchvision.dataset.MNIST)
        """

        # Use fixed seed (0 for test set, 1 for training set)
        torch.manual_seed(int(train))

        # Sample radians
        rad = np.radians(degree)
        thetas = (torch.rand(len(self.data)) * 2 - 1) * rad

        # Create rotation matrices
        c, s = torch.cos(thetas), torch.sin(thetas)
        rot_matrices = torch.stack((torch.stack((c, -s, c*0), 0),
                                    torch.stack((s, c, s*0), 0)), 0).permute(2, 0, 1)

        # Create resampling grid
        rot_grids = F.affine_grid(rot_matrices, self.data.unsqueeze(1).shape, align_corners=False)

        # Rotate images using 'mode' interpolation (e.g. bilinear sampling)
        xmin, xmax = self.data.min(), self.data.max()
        data_dtype = self.data.dtype

        self.data = F.grid_sample(self.data.unsqueeze(1).float(), rot_grids, align_corners=False, mode=mode)
        self.data = self.data[:, 0].type(data_dtype).clamp(xmin, xmax)


class TranslatedMNIST(datasets.MNIST):
    """ MNIST translated by fixed amount using random seed """

    def __init__(self, root: str, amount: float = 8.0, mode: str = 'bilinear',
                 train: bool = True, transform: Union[Callable, type(None)] = None,
                 target_transform: Union[Callable, type(None)] = None, download: bool = False):
        super().__init__(root, train, transform, target_transform, download)
        """
        Args:
            root (string): Root directory of dataset where ``MNIST/processed/training.pt``
                and  ``MNIST/processed/test.pt`` exist.
            amount (float): Amount of translation in pixels
            mode (float): Mode used for interpolation (nearest|bilinear|bicubic)
            train (bool, optional): If True, creates dataset from ``training.pt``,
                otherwise from ``test.pt``.
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
                same arguments as torchvision.see arguments of torchvision.dataset.MNIST)
        """

        torch.manual_seed(int(train))

        rad = np.radians(amount)

        r1 = (torch.rand(len(self.data)) * 2 - 1) * amount / 14.0
        r2 = (torch.rand(len(self.data)) * 2 - 1) * amount / 14.0

        zero = r1 * 0.0
        one = zero + 1.0

        matrices = torch.stack((torch.stack((one, zero, r1), 0),
                                torch.stack((zero, one, r2), 0)), 0).permute(2, 0, 1)

        grids = F.affine_grid(matrices, self.data.unsqueeze(1).shape, align_corners=True)

        xmin, xmax = self.data.min(), self.data.max()
        data_dtype = self.data.dtype

        self.data = F.grid_sample(self.data.unsqueeze(1).float(), grids, align_corners=True, mode=mode)
        self.data = self.data[:, 0].type(data_dtype).clamp(xmin, xmax)


class ScaledMNIST(datasets.MNIST):
    """ MNIST scaled by fixed amount using random seed """

    def __init__(self, root: str, amount: float = LOG2, mode: str = 'bilinear',
                 train: bool = True, transform: Union[Callable, type(None)] = None,
                 target_transform: Union[Callable, type(None)] = None, download: bool = False):
        super().__init__(root, train, transform, target_transform, download)
        """
        Args:
            root (string): Root directory of dataset where ``MNIST/processed/training.pt``
                and  ``MNIST/processed/test.pt`` exist.
            amount (float): Amount of translation in pixels
            mode (float): Mode used for interpolation (nearest|bilinear|bicubic)
            train (bool, optional): If True, creates dataset from ``training.pt``,
                otherwise from ``test.pt``.
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
                same arguments as torchvision.see arguments of torchvision.dataset.MNIST)
        """
        torch.manual_seed(int(train))
        s1 = (torch.rand(len(self.data)) * 2 - 1) * amount

        zero = s1 * 0.0

        matrices = torch.stack((torch.stack((torch.exp(s1), zero, zero), 0),
                                torch.stack((zero, torch.exp(s1), zero), 0)), 0).permute(2, 0, 1)

        grids = F.affine_grid(matrices, self.data.unsqueeze(1).shape, align_corners=True)

        xmin, xmax = self.data.min(), self.data.max()
        data_dtype = self.data.dtype

        self.data = F.grid_sample(self.data.unsqueeze(1).float(), grids, align_corners=True, mode=mode)
        self.data = self.data[:, 0].type(data_dtype).clamp(xmin, xmax)


class RotatedFashionMNIST(datasets.FashionMNIST):
    """ Rotated FashionMNIST class.
        Wraps regular pytorch FashionMNIST and rotates each image randomly between given angle using fixed seed. """

    def __init__(self, root: str, degree: float, mode: str = 'bilinear',
                 train: bool = True, transform: Union[Callable, type(None)] = None,
                 target_transform: Union[Callable, type(None)] = None, download: bool = False):
        super().__init__(root, train, transform, target_transform, download)
        """
        Args:
            root (string): Root directory of dataset where ``FashionMNIST/processed/training.pt``
                and  ``FashionMNIST/processed/test.pt`` exist.
            degree (float): Amount of rotation in degrees
            mode (float): Mode used for interpolation (nearest|bilinear|bicubic)
            train (bool, optional): If True, creates dataset from ``training.pt``,
                otherwise from ``test.pt``.
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
                same arguments as torchvision.see arguments of torchvision.dataset.FashionMNIST)
        """
        # Use fixed seed (0 for test set, 1 for training set)
        torch.manual_seed(int(train))

        # Sample radians
        rad = np.radians(degree)
        thetas = (torch.rand(len(self.data)) * 2 - 1) * rad

        # Create rotation matrices
        c, s = torch.cos(thetas), torch.sin(thetas)
        rot_matrices = torch.stack((torch.stack((c, -s, c*0), 0),
                                    torch.stack((s, c, s*0), 0)), 0).permute(2, 0, 1)

        # Create resampling grid
        rot_grids = F.affine_grid(rot_matrices, self.data.unsqueeze(1).shape, align_corners=False)

        # Rotate images using 'mode' interpolation (e.g. bilinear sampling)
        xmin, xmax = self.data.min(), self.data.max()
        data_dtype = self.data.dtype

        self.data = F.grid_sample(self.data.unsqueeze(1).float(), rot_grids, align_corners=False, mode=mode)
        self.data = self.data[:, 0].type(data_dtype).clamp(xmin, xmax)


class RotatedFashionMNIST(datasets.FashionMNIST):
    """ Rotated Fashion MNIST dataset.
        Wraps regular pytorch FashionMNIST and rotates each image randomly between given angle using fixed seed. """

    def __init__(self, root: str, degree: float, mode: str = 'bilinear',
                 train: bool = True, transform: Union[Callable, type(None)] = None,
                 target_transform: Union[Callable, type(None)] = None, download: bool = False):
        super().__init__(root, train, transform, target_transform, download)
        """
        Args:
            root (string): Root directory of dataset where ``FashionMNIST/processed/training.pt``
                and  ``FashionMNIST/processed/test.pt`` exist.
            degree (float): Amount of rotation in degrees
            mode (float): Mode used for interpolation (nearest|bilinear|bicubic)
            train (bool, optional): If True, creates dataset from ``training.pt``,
                otherwise from ``test.pt``.
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
                same arguments as torchvision.see arguments of torchvision.dataset.FashionMNIST)
        """
        # Use fixed seed (0 for test set, 1 for training set)
        torch.manual_seed(int(train))

        # Sample radians
        rad = np.radians(degree)
        thetas = (torch.rand(len(self.data)) * 2 - 1) * rad

        # Create rotation matrices
        c, s = torch.cos(thetas), torch.sin(thetas)
        rot_matrices = torch.stack((torch.stack((c, -s, c*0), 0),
                                    torch.stack((s, c, s*0), 0)), 0).permute(2, 0, 1)

        # Create resampling grid
        rot_grids = F.affine_grid(rot_matrices, self.data.unsqueeze(1).shape, align_corners=False)

        # Rotate images using 'mode' interpolation (e.g. bilinear sampling)
        xmin, xmax = self.data.min(), self.data.max()
        data_dtype = self.data.dtype

        self.data = F.grid_sample(self.data.unsqueeze(1).float(), rot_grids, align_corners=False, mode=mode)
        self.data = self.data[:, 0].type(data_dtype).clamp(xmin, xmax)


class TranslatedFashionMNIST(datasets.FashionMNIST):
    """ FashionMNIST translated by fixed amount using random seed """

    def __init__(self, root: str, amount: float = 8.0, mode: str = 'bilinear',
                 train: bool = True, transform: Union[Callable, type(None)] = None,
                 target_transform: Union[Callable, type(None)] = None, download: bool = False):
        super().__init__(root, train, transform, target_transform, download)
        """
        Args:
            root (string): Root directory of dataset where ``FashionMNIST/processed/training.pt``
                and  ``FashionMNIST/processed/test.pt`` exist.
            amount (float): Amount of translation in pixels
            mode (float): Mode used for interpolation (nearest|bilinear|bicubic)
            train (bool, optional): If True, creates dataset from ``training.pt``,
                otherwise from ``test.pt``.
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
                same arguments as torchvision.see arguments of torchvision.dataset.FashionMNIST)
        """
        torch.manual_seed(int(train))
        r1 = (torch.rand(len(self.data)) * 2 - 1) * amount / 14.0
        r2 = (torch.rand(len(self.data)) * 2 - 1) * amount / 14.0

        zero = r1 * 0.0
        one = zero + 1.0

        matrices = torch.stack((torch.stack((one, zero, r1), 0),
                                torch.stack((zero, one, r2), 0)), 0).permute(2, 0, 1)

        grids = F.affine_grid(matrices, self.data.unsqueeze(1).shape, align_corners=True)

        xmin, xmax = self.data.min(), self.data.max()
        data_dtype = self.data.dtype

        self.data = F.grid_sample(self.data.unsqueeze(1).float(), grids, align_corners=True, mode=mode)
        self.data = self.data[:, 0].type(data_dtype).clamp(xmin, xmax)


class ScaledFashionMNIST(datasets.FashionMNIST):
    """ FashionMNIST scaled by fixed amount using random seed """

    def __init__(self, root: str, amount: float = LOG2, mode: str = 'bilinear',
                 train: bool = True, transform: Union[Callable, type(None)] = None,
                 target_transform: Union[Callable, type(None)] = None, download: bool = False):
        super().__init__(root, train, transform, target_transform, download)
        """
        Args:
            root (string): Root directory of dataset where ``FashionMNIST/processed/training.pt``
                and  ``FashionMNIST/processed/test.pt`` exist.
            amount (float): Amount of translation in pixels
            mode (float): Mode used for interpolation (nearest|bilinear|bicubic)
            train (bool, optional): If True, creates dataset from ``training.pt``,
                otherwise from ``test.pt``.
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
                same arguments as torchvision.see arguments of torchvision.dataset.FashionMNIST)
        """

        torch.manual_seed(int(train))

        s1 = (torch.rand(len(self.data)) * 2 - 1) * amount
        zero = s1 * 0.0
        matrices = torch.stack((torch.stack((torch.exp(s1), zero, zero), 0),
                                torch.stack((zero, torch.exp(s1), zero), 0)), 0).permute(2, 0, 1)

        grids = F.affine_grid(matrices, self.data.unsqueeze(1).shape, align_corners=True)

        xmin, xmax = self.data.min(), self.data.max()
        data_dtype = self.data.dtype

        self.data = F.grid_sample(self.data.unsqueeze(1).float(), grids, align_corners=True, mode=mode)
        self.data = self.data[:, 0].type(data_dtype).clamp(xmin, xmax)


class RotatedCIFAR10(datasets.CIFAR10):
    """ CIFAR10 rotated by fixed amount using random seed """

    def __init__(self, root: str, degree: float, mode: str = 'bilinear',
                 train: bool = True, transform: Union[Callable, type(None)] = None,
                 target_transform: Union[Callable, type(None)] = None, download: bool = False):
        super().__init__(root, train, transform, target_transform, download)
        """
        Args:
            root (string): Root directory of dataset where ``CIFAR10/processed/training.pt``
                and  ``CIFAR10/processed/test.pt`` exist.
            degree (float): Amount of rotation in degrees
            mode (float): Mode used for interpolation (nearest|bilinear|bicubic)
            train (bool, optional): If True, creates dataset from ``training.pt``,
                otherwise from ``test.pt``.
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
                same arguments as torchvision.see arguments of torchvision.dataset.CIFAR10)
        """

        torch.manual_seed(int(train))

        rad = np.radians(degree)

        thetas = (torch.rand(len(self.data)) * 2 - 1) * rad

        c, s = torch.cos(thetas), torch.sin(thetas)

        rot_matrices = torch.stack((torch.stack((c, -s, c*0), 0),
                                    torch.stack((s, c, s*0), 0)), 0).permute(2, 0, 1)

        xmin, xmax = self.data.min(), self.data.max()
        data_dtype = self.data.dtype

        self.data = torch.Tensor(self.data.transpose(0, 3, 1, 2))
        rot_grids = F.affine_grid(rot_matrices, self.data.shape, align_corners=False)

        self.data = F.grid_sample(self.data.float(), rot_grids, align_corners=False, mode=mode)
        self.data = self.data.clamp(xmin, xmax).numpy().astype(data_dtype)
        self.data = self.data.transpose(0, 2, 3, 1)


class TranslatedCIFAR10(datasets.CIFAR10):
    """ CIFAR10 translated by fixed amount using random seed """

    def __init__(self, root: str, amount: float = 8.0, mode: str = 'bilinear',
                 train: bool = True, transform: Union[Callable, type(None)] = None,
                 target_transform: Union[Callable, type(None)] = None, download: bool = False):
        super().__init__(root, train, transform, target_transform, download)
        """
        Args:
            root (string): Root directory of dataset where ``CIFAR10/processed/training.pt``
                and  ``CIFAR10/processed/test.pt`` exist.
            amount (float): Amount of translation in pixels
            mode (float): Mode used for interpolation (nearest|bilinear|bicubic)
            train (bool, optional): If True, creates dataset from ``training.pt``,
                otherwise from ``test.pt``.
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
                same arguments as torchvision.see arguments of torchvision.dataset.CIFAR10)
        """

        torch.manual_seed(int(train))

        rad = np.radians(amount)

        r1 = (torch.rand(len(self.data)) * 2 - 1) * amount / 14.0
        r2 = (torch.rand(len(self.data)) * 2 - 1) * amount / 14.0

        zero = r1 * 0.0
        one = zero + 1.0

        matrices = torch.stack((torch.stack((one, zero, r1), 0),
                                torch.stack((zero, one, r2), 0)), 0).permute(2, 0, 1)

        xmin, xmax = self.data.min(), self.data.max()
        data_dtype = self.data.dtype

        self.data = torch.Tensor(self.data.transpose(0, 3, 1, 2))
        grids = F.affine_grid(matrices, self.data.shape, align_corners=False)

        self.data = F.grid_sample(self.data.float(), grids, align_corners=False, mode=mode)
        self.data = self.data.clamp(xmin, xmax).numpy().astype(data_dtype)
        self.data = self.data.transpose(0, 2, 3, 1)


class ScaledCIFAR10(datasets.CIFAR10):
    """ CIFAR10 scaled by fixed amount using random seed """

    def __init__(self, root: str, amount: float = LOG2, mode: str = 'bilinear',
                 train: bool = True, transform: Union[Callable, type(None)] = None,
                 target_transform: Union[Callable, type(None)] = None, download: bool = False):
        super().__init__(root, train, transform, target_transform, download)
        """
        Args:
            root (string): Root directory of dataset where ``CIFAR10/processed/training.pt``
                and  ``CIFAR10/processed/test.pt`` exist.
            amount (float): Amount of translation in pixels
            mode (float): Mode used for interpolation (nearest|bilinear|bicubic)
            train (bool, optional): If True, creates dataset from ``training.pt``,
                otherwise from ``test.pt``.
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
                same arguments as torchvision.see arguments of torchvision.dataset.CIFAR10)
        """

        torch.manual_seed(int(train))

        rad = np.radians(amount)

        s1 = (torch.rand(len(self.data)) * 2 - 1) * amount

        zero = s1 * 0.0
        one = zero + 1.0

        matrices = torch.stack((torch.stack((torch.exp(s1), zero, zero), 0),
                                torch.stack((zero, torch.exp(s1), zero), 0)), 0).permute(2, 0, 1)

        xmin, xmax = self.data.min(), self.data.max()
        data_dtype = self.data.dtype

        self.data = torch.Tensor(self.data.transpose(0, 3, 1, 2))
        grids = F.affine_grid(matrices, self.data.shape, align_corners=False)

        self.data = F.grid_sample(self.data.float(), grids, align_corners=False, mode=mode)
        self.data = self.data.clamp(xmin, xmax).numpy().astype(data_dtype)
        self.data = self.data.transpose(0, 2, 3, 1)

class RotatedCIFAR100(datasets.CIFAR100):
    """ CIFAR100 rotated by fixed amount using random seed """

    def __init__(self, root: str, degree: float, mode: str = 'bilinear',
                 train: bool = True, transform: Union[Callable, type(None)] = None,
                 target_transform: Union[Callable, type(None)] = None, download: bool = False):
        super().__init__(root, train, transform, target_transform, download)
        """
        Args:
            root (string): Root directory of dataset where ``CIFAR100/processed/training.pt``
                and  ``CIFAR100/processed/test.pt`` exist.
            degree (float): Amount of rotation in degrees
            mode (float): Mode used for interpolation (nearest|bilinear|bicubic)
            train (bool, optional): If True, creates dataset from ``training.pt``,
                otherwise from ``test.pt``.
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
                same arguments as torchvision.see arguments of torchvision.dataset.CIFAR100)
        """

        torch.manual_seed(int(train))

        rad = np.radians(degree)

        thetas = (torch.rand(len(self.data)) * 2 - 1) * rad

        c, s = torch.cos(thetas), torch.sin(thetas)

        rot_matrices = torch.stack((torch.stack((c, -s, c*0), 0),
                                    torch.stack((s, c, s*0), 0)), 0).permute(2, 0, 1)

        xmin, xmax = self.data.min(), self.data.max()
        data_dtype = self.data.dtype

        self.data = torch.Tensor(self.data.transpose(0, 3, 1, 2))
        rot_grids = F.affine_grid(rot_matrices, self.data.shape, align_corners=False)

        self.data = F.grid_sample(self.data.float(), rot_grids, align_corners=False, mode=mode)
        self.data = self.data.clamp(xmin, xmax).numpy().astype(data_dtype)
        self.data = self.data.transpose(0, 2, 3, 1)


class TranslatedCIFAR100(datasets.CIFAR100):
    """ CIFAR100 translated by fixed amount using random seed """

    def __init__(self, root: str, amount: float = 8.0, mode: str = 'bilinear',
                 train: bool = True, transform: Union[Callable, type(None)] = None,
                 target_transform: Union[Callable, type(None)] = None, download: bool = False):
        super().__init__(root, train, transform, target_transform, download)
        """
        Args:
            root (string): Root directory of dataset where ``CIFAR100/processed/training.pt``
                and  ``CIFAR100/processed/test.pt`` exist.
            amount (float): Amount of translation in pixels
            mode (float): Mode used for interpolation (nearest|bilinear|bicubic)
            train (bool, optional): If True, creates dataset from ``training.pt``,
                otherwise from ``test.pt``.
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
                same arguments as torchvision.see arguments of torchvision.dataset.CIFAR100)
        """

        torch.manual_seed(int(train))

        rad = np.radians(amount)

        r1 = (torch.rand(len(self.data)) * 2 - 1) * amount / 14.0
        r2 = (torch.rand(len(self.data)) * 2 - 1) * amount / 14.0

        zero = r1 * 0.0
        one = zero + 1.0

        matrices = torch.stack((torch.stack((one, zero, r1), 0),
                                torch.stack((zero, one, r2), 0)), 0).permute(2, 0, 1)

        xmin, xmax = self.data.min(), self.data.max()
        data_dtype = self.data.dtype

        self.data = torch.Tensor(self.data.transpose(0, 3, 1, 2))
        grids = F.affine_grid(matrices, self.data.shape, align_corners=False)

        self.data = F.grid_sample(self.data.float(), grids, align_corners=False, mode=mode)
        self.data = self.data.clamp(xmin, xmax).numpy().astype(data_dtype)
        self.data = self.data.transpose(0, 2, 3, 1)


class ScaledCIFAR100(datasets.CIFAR100):
    """ CIFAR100 scaled by fixed amount using random seed """

    def __init__(self, root: str, amount: float = LOG2, mode: str = 'bilinear',
                 train: bool = True, transform: Union[Callable, type(None)] = None,
                 target_transform: Union[Callable, type(None)] = None, download: bool = False):
        super().__init__(root, train, transform, target_transform, download)
        """
        Args:
            root (string): Root directory of dataset where ``CIFAR100/processed/training.pt``
                and  ``CIFAR100/processed/test.pt`` exist.
            amount (float): Amount of translation in pixels
            mode (float): Mode used for interpolation (nearest|bilinear|bicubic)
            train (bool, optional): If True, creates dataset from ``training.pt``,
                otherwise from ``test.pt``.
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
                same arguments as torchvision.see arguments of torchvision.dataset.CIFAR100)
        """

        torch.manual_seed(int(train))

        rad = np.radians(amount)

        s1 = (torch.rand(len(self.data)) * 2 - 1) * amount

        zero = s1 * 0.0
        one = zero + 1.0

        matrices = torch.stack((torch.stack((torch.exp(s1), zero, zero), 0),
                                torch.stack((zero, torch.exp(s1), zero), 0)), 0).permute(2, 0, 1)

        xmin, xmax = self.data.min(), self.data.max()
        data_dtype = self.data.dtype

        self.data = torch.Tensor(self.data.transpose(0, 3, 1, 2))
        grids = F.affine_grid(matrices, self.data.shape, align_corners=False)

        self.data = F.grid_sample(self.data.float(), grids, align_corners=False, mode=mode)
        self.data = self.data.clamp(xmin, xmax).numpy().astype(data_dtype)
        self.data = self.data.transpose(0, 2, 3, 1)
