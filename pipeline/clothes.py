import torchvision
from predictors.YOLOv3 import YOLOv3Predictor
import torch
import os
import cv2
from yolo.utils.utils import *
from predictors.YOLOv3 import YOLOv3Predictor
import glob
from tqdm import tqdm
import sys

SIMPLE_CROP_W = 200
SIMPLE_CROP_H = 200


class ClothesDetectorYOLOv3():

    # localize clothes bbox based on YOLOv3 clothes detector
    # source: https://github.com/AlberetOZ/WondeRobe_Clothes_test

    def __init__(self, params: dict) -> None:
        '''desc

        :param params: 
        :return: 
        '''
        self.params = params
        self.classes = load_classes(self.params["class_path"])

        self.detectron = YOLOv3Predictor(params=self.params)

    
    def process(self, img: np.ndarray) -> np.ndarray:
        '''desc
        
        :param img:
        :return: 
        '''
        result = self.detectron.get_detections(img)

        return np.array(result)


def clothes_bbox(img: np.ndarray) -> np.ndarray:
    '''desc
    
    :param img:
    :return:
    '''
    # bbox based on tips from the initial note

    w, h = img.shape[:2]

    xmin = w//2 - SIMPLE_CROP_W//2
    xmin = max(xmin, 0)

    xmax = w//2 + SIMPLE_CROP_W//2
    xmax = min(xmax, w)

    ymin = h//2 - SIMPLE_CROP_H//2
    ymin = max(ymin, 0)

    ymax = h//2 + SIMPLE_CROP_H//2
    ymax = min(ymax, h)

    result = np.array([[xmin, ymin, xmax, ymax, 1, -1]])

    return result

class CropClothes(object):
    # Custom transform to crop image based on bbox of object
    def __init__(self, bbox_funct) -> None:
        self.bbox_funct = bbox_funct

    def __call__(self, sample: np.ndarray) -> np.ndarray:
        
        w, h = sample.shape[:2]

        candidates = self.bbox_funct(sample)

        # print(candidates)
        
        

        if len(candidates) == 0:
            # if network based bbox is not found
            # bbox will be centered box from tips
            # print('ZERO')
            candidates = clothes_bbox(sample)
        
        candidates[np.where(candidates[:, 0:2] < 0)] = 0
        candidates[np.where(candidates[:, 2] > 256), 2] = 256
        candidates[np.where(candidates[:, 3] > 256), 3] = 256

        # print(candidates)

        center = np.array([[w//2, h//2]])
        # print('image center')
        # print(center)

        candidate_centers = candidates[:, 2:4] - candidates[:, 0:2]
        # print('cand centersraw')
        # print(candidate_centers)
        # print('cand centers')
        candidate_centers = np.divide(candidate_centers, 2, casting='unsafe') + candidates[:, 0:2]
        # print(candidate_centers)

        distances = np.linalg.norm(candidate_centers - center, axis=1)
        # print('dist')
        # print(distances)

        out = candidates[np.argmin(distances)]
        # print('out')
        # print(out)
        xmin, ymin, xmax, ymax, *_ = map(int, out)

        crop = sample[ymin:ymax, xmin:xmax, :]

        return crop


class TrainDatasetFolder(torchvision.datasets.VisionDataset):

    def __init__(
            self,
            root: str,
            loader,
            extensions=None,
            transform=None,
            target_transform=None,
            is_valid_file=None,
    ) -> None:
        super(TrainDatasetFolder, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        classes, class_to_idx = self._find_classes(self.root)
        samples = self.make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
    
    def make_dataset(self, directory: str,
        class_to_idx,
        extensions,
        is_valid_file):

        if class_to_idx is None:
            # prevent potential bug since make_dataset() would use the class_to_idx logic of the
            # find_classes() function, instead of using that of the find_classes() method, which
            # is potentially overridden and thus could have a different logic.
            raise ValueError(
                "The class_to_idx parameter cannot be None."
            )
        
        instances = []
        available_classes = set()
        for subdir, target_class in self.subdir_to_class.items():
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(directory, subdir)
            if not os.path.isdir(target_dir):
                continue
            
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    
                    if is_valid_file is None or is_valid_file(path):
                        item = path, class_index
                        instances.append(item)

                        if target_class not in available_classes:
                            available_classes.add(target_class)
                    

        
        empty_classes = set(class_to_idx.keys()) - available_classes
        if empty_classes:
            msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
            if extensions is not None:
                msg += f"Supported extensions are: {', '.join(extensions)}"
            raise FileNotFoundError(msg)
        
        return instances
    
    def _find_classes(self, dir: str):
        
        subdirs = sorted(entry.name for entry in os.scandir(dir) if entry.is_dir())
        if not subdirs:
            raise FileNotFoundError(f"Couldn't find any subdirs in {dir}.")
        
        class_to_idx = {}
        self.subdir_to_class = {}
        for subdir in subdirs:
            
            sample = next(os.scandir(f'{dir}/{subdir}'))

            *_, current_class = sample.name.split('_', 2)
            current_class, *_ = current_class.rsplit('_', 5)
            self.subdir_to_class[subdir] = current_class
            if current_class not in class_to_idx.keys():
                class_to_idx[current_class] = len(class_to_idx)

        return list(class_to_idx.keys()), class_to_idx
    
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)