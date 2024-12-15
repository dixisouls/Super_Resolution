import torch
from PIL import Image
import argparse
from torchvision.transforms import transforms
from models.edsr_channel_attention import EDSR
from config import Config
from utils.infer_utils import setup_logging
import time
import warnings
import os

warnings.filterwarnings("ignore")


def inference(path_to_image):
    config = Config()

    # setup logging
    logger = setup_logging(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # record start time
    start_time = time.time()

    logger.info(f"Inference Started")

    # Load the model
    model = EDSR(
        config.SCALE_FACTOR,
        config.NUM_CHANNELS,
        config.NUM_FEATURES,
        config.NUM_RES_BLOCKS,
        config.REDUCTION_RATIO,
    ).to(device)
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
    model.eval()
    logger.info(f"Model loaded from {config.MODEL_PATH}")

    # preprocess the image
    image = Image.open(path_to_image).convert("RGB")
    width, height = image.size
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(
                (height, width), interpolation=transforms.InterpolationMode.BICUBIC
            ),
        ]
    )
    image_tensor = transform(image).unsqueeze(0).to(device)
    logger.info(f"Image loaded from {path_to_image}")

    # Perform inference
    with torch.no_grad():
        output_tensor = model(image_tensor)

    # postprocess the image
    output_tensor = output_tensor.squeeze(0).cpu().clamp(0, 1)
    output_image = transforms.ToPILImage()(output_tensor)

    # Save the image
    save_path = os.path.join(os.path.dirname(path_to_image), "output.png")
    output_image.save(save_path)

    logger.info(f"Inference Completed")

    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"Total time taken: {total_time:.2f} seconds")

    logger.info(f"Output image saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on an image.")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    args = parser.parse_args()

    inference(args.image_path)
