from vigc.common.registry import registry
from omegaconf import OmegaConf
import albumentations as alb
from albumentations.pytorch import ToTensorV2
from vigc.processors.base_processor import BaseProcessor
import numpy as np
import cv2
from PIL import Image, ImageOps
from torchvision.transforms.functional import resize


class FormulaImageBaseProcessor(BaseProcessor):

    def __init__(self, image_size):
        super(FormulaImageBaseProcessor, self).__init__()
        self.input_size = [image_size, image_size]

    @staticmethod
    def crop_margin(img: Image.Image) -> Image.Image:
        data = np.array(img.convert("L"))
        data = data.astype(np.uint8)
        max_val = data.max()
        min_val = data.min()
        if max_val == min_val:
            return img
        data = (data - min_val) / (max_val - min_val) * 255
        gray = 255 * (data < 200).astype(np.uint8)

        coords = cv2.findNonZero(gray)  # Find all non-zero points (text)
        a, b, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
        return img.crop((a, b, w + a, h + b))

    def prepare_input(self, img: Image.Image, random_padding: bool = False):
        """
        Convert PIL Image to tensor according to specified input_size after following steps below:
            - resize
            - rotate (if align_long_axis is True and image is not aligned longer axis with canvas)
            - pad
        """
        if img is None:
            return
        # crop margins
        try:
            img = self.crop_margin(img.convert("RGB"))
        except OSError:
            # might throw an error for broken files
            return
        if img.height == 0 or img.width == 0:
            return

        img = resize(img, min(self.input_size))
        img.thumbnail((self.input_size[1], self.input_size[0]))
        delta_width = self.input_size[1] - img.width
        delta_height = self.input_size[0] - img.height
        if random_padding:
            pad_width = np.random.randint(low=0, high=delta_width + 1)
            pad_height = np.random.randint(low=0, high=delta_height + 1)
        else:
            pad_width = delta_width // 2
            pad_height = delta_height // 2
        padding = (
            pad_width,
            pad_height,
            delta_width - pad_width,
            delta_height - pad_height,
        )
        return ImageOps.expand(img, padding)


@registry.register_processor("formula_image_train")
class FormulaImageTrainProcessor(FormulaImageBaseProcessor):
    def __init__(self, image_size=384, random_padding=False):
        super().__init__(image_size)
        self.random_padding = random_padding

        self.transform = alb.Compose(
            [
                alb.Compose(
                    [alb.ShiftScaleRotate(shift_limit=0, scale_limit=(-.15, 0), rotate_limit=1, border_mode=0,
                                          interpolation=3,
                                          value=[255, 255, 255], p=1),
                     alb.GridDistortion(distort_limit=0.1, border_mode=0, interpolation=3, value=[255, 255, 255],
                                        p=.5)], p=.15),
                # alb.InvertImg(p=.15),
                alb.RGBShift(r_shift_limit=15, g_shift_limit=15,
                             b_shift_limit=15, p=0.3),
                alb.GaussNoise(10, p=.2),
                alb.RandomBrightnessContrast(.05, (-.2, 0), True, p=0.2),
                alb.ImageCompression(95, p=.3),
                alb.ToGray(always_apply=True),
                alb.Normalize((0.7931, 0.7931, 0.7931), (0.1738, 0.1738, 0.1738)),
                # alb.Sharpen()
                ToTensorV2(),
            ]
        )

    def __call__(self, item):
        img = self.prepare_input(item, self.random_padding)
        if img is None:
            return img
        return self.transform(img)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 384)
        random_padding = cfg.get("random_padding", False)

        return cls(
            image_size=image_size,
            random_padding=random_padding
        )


@registry.register_processor("formula_image_eval")
class FormulaImageEvalProcessor(FormulaImageBaseProcessor):
    def __init__(self, image_size=384):
        super().__init__(image_size)

        self.transform = alb.Compose(
            [
                alb.ToGray(always_apply=True),
                alb.Normalize((0.7931, 0.7931, 0.7931), (0.1738, 0.1738, 0.1738)),
                # alb.Sharpen()
                ToTensorV2(),
            ]
        )

    def __call__(self, item):
        image = self.prepare_input(item)
        return self.transform(image)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 384)

        return cls(image_size=image_size)