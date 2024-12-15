import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class SuperResolutionDataset(Dataset):
    def __init__(self, high_res_dir, low_res_dir, scale_factor):
        self.high_res_filenames = [
            os.path.join(high_res_dir, f)
            for f in os.listdir(high_res_dir)
            if os.path.isfile(os.path.join(high_res_dir, f))
        ]
        self.low_res_filenames = [
            os.path.join(low_res_dir, f)
            for f in os.listdir(low_res_dir)
            if os.path.isfile(os.path.join(low_res_dir, f))
        ]
        self.scale_factor = scale_factor

        # ensure the lists are aligned
        assert len(self.high_res_filenames) == len(
            self.low_res_filenames
        )  # check if the number of high res and low res images are the same

        self.transform_low_res = transforms.Compose(
            [
                transforms.Resize(
                    (512 // scale_factor, 512 // scale_factor), Image.BICUBIC
                ),
                transforms.ToTensor(),
            ]
        )

        self.transform_high_res = transforms.Compose(
            [
                transforms.Resize((512, 512), Image.BICUBIC),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.high_res_filenames)

    def __getitem__(self, idx):
        high_res_image = Image.open(self.high_res_filenames[idx])
        low_res_image = Image.open(self.low_res_filenames[idx])

        # convert to RGB if not already
        if high_res_image.mode != "RGB":
            high_res_image = high_res_image.convert("RGB")
        if low_res_image.mode != "RGB":
            low_res_image = low_res_image.convert("RGB")

        return self.transform_low_res(low_res_image), self.transform_high_res(
            high_res_image
        )


def get_data_loader(high_res_dir, low_res_dir, scale_factor, batch_size):
    dataset = SuperResolutionDataset(high_res_dir, low_res_dir, scale_factor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader
