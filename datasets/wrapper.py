import torchvision.transforms as T
from torch.utils.data import Dataset as TorchDataset
from PIL import Image
from torchvision.transforms import InterpolationMode

from dassl.utils import read_image

INTERPOLATION_MODES = {
    'bilinear': InterpolationMode.BILINEAR,
    'bicubic': InterpolationMode.BICUBIC,
    'nearest': InterpolationMode.NEAREST
}


class DatasetWrapper(TorchDataset):

    def __init__(self, cfg, data_source, transform=None, is_train=False):
        self.cfg = cfg
        self.data_source = data_source
        self.transform = transform  # accept list (tuple) as input
        self.is_train = is_train
        # Augmenting an image K>1 times is only allowed during training
        self.k_tfm = cfg.DATALOADER.K_TRANSFORMS if is_train else 1
        self.return_img0 = cfg.DATALOADER.RETURN_IMG0

        if self.k_tfm > 1 and transform is None:
            raise ValueError(
                'Cannot augment the image {} times '
                'because transform is None'.format(self.k_tfm)
            )

        # Build transform that doesn't apply any data augmentation
        interp_mode = INTERPOLATION_MODES[cfg.INPUT.INTERPOLATION]
        to_tensor = []
        to_tensor += [T.Resize(cfg.INPUT.SIZE, interpolation=interp_mode)]
        to_tensor += [T.ToTensor()]
        if 'normalize' in cfg.INPUT.TRANSFORMS:
            normalize = T.Normalize(
                mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
            )
            to_tensor += [normalize]
        self.to_tensor = T.Compose(to_tensor)

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]

        output = {
            'label': item.label,
            'domain': item.domain,
            'impath': item.impath
        }

        img0 = read_image(item.impath)

        if self.transform is not None:
            if isinstance(self.transform, (list, tuple)):
                for i, tfm in enumerate(self.transform):
                    img = self._transform_image(tfm, img0)
                    keyname = 'img'
                    if (i + 1) > 1:
                        keyname += str(i + 1)
                    output[keyname] = img
            else:
                img = self._transform_image(self.transform, img0)
                output['img'] = img

        if self.return_img0:
            output['img0'] = self.to_tensor(img0)

        return output

    def _transform_image(self, tfm, img0):
        img_list = []

        for k in range(self.k_tfm):
            img_list.append(tfm(img0))

        img = img_list
        if len(img) == 1:
            img = img[0]

        return img
