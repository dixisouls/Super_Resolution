import torch
import torch.nn as nn
import torch.optim as optim
from utils.data_utils import get_data_loader
from models.edsr_channel_attention import EDSR
from config import Config
from utils.utils import setup_logging
import os
from tqdm import tqdm
import time


def clear_logs():
    try:
        os.remove("logs/train.log")
    except FileNotFoundError:
        pass


def train():
    config = Config()

    # clear logs
    clear_logs()

    # setup logging
    logger = setup_logging(config, config.TRAIN_LOG)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Initialize the model, loss function and optimizer
    model = EDSR(
        config.SCALE_FACTOR,
        config.NUM_CHANNELS,
        config.NUM_FEATURES,
        config.NUM_RES_BLOCKS,
        config.REDUCTION_RATIO
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    logger.info(f"Model: {config.MODEL_NAME}")
    logger.info(f"Learning rate: {config.LEARNING_RATE}")

    #log number of parameters
    logger.info(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    # Get the data loader
    train_loader = get_data_loader(
        config.HIGH_RES_DIR, config.LOW_RES_DIR, config.SCALE_FACTOR, config.BATCH_SIZE
    )

    # create the output directory if it does not exist
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # training loop
    for epoch in range(config.EPOCHS):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{config.EPOCHS}", leave=False
        )

        for batch_idx, (low_res, high_res) in enumerate(progress_bar):
            low_res, high_res = low_res.to(device), high_res.to(device)

            # forward pass
            outputs = model(low_res)
            loss = criterion(outputs, high_res)

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # log the loss and CUDA memory usage
            progress_bar.set_postfix(
                {
                    "training_loss": f"{loss.item():.4f}",
                    "cuda_mem_allocated": f"{torch.cuda.memory_allocated(device) / 1024 ** 2:.2f} MB",
                }
            )
            logger.debug(
                f"Epoch: {epoch + 1}, Batch: {batch_idx + 1}, Loss: {loss.item():.4f}"
            )

        # log epoch statistics
        epoch_loss = running_loss / len(train_loader)
        epoch_duration = (time.time() - start_time)/60
        logger.info(
            f"Epoch: {epoch + 1}/{config.EPOCHS}, Loss: {epoch_loss:.4f}, Duration: {epoch_duration:.2f} minutes"
        )

        # save the model
        if (epoch + 1) % config.SAVE_MODEL_EVERY == 0 or epoch + 1 == config.EPOCHS:
            model_path = os.path.join(
                config.OUTPUT_DIR, f"{config.MODEL_NAME}_epoch_{epoch + 1}.pth"
            )
            torch.save(model.state_dict(), model_path)
            logger.info(f"Saved model to {model_path}")


if __name__ == "__main__":
    train()
