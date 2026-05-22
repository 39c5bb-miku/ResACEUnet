import os
import torch


def load_checkpoint(config, model, optimizer, lr_scheduler, logger):
    logger.info(
        f"==============> Resuming form {config.model.resume}...................."
    )
    if config.model.resume.startswith("https"):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.model.resume, map_location="cpu", check_hash=True, weights_only=False
        )
    else:
        checkpoint = torch.load(
            config.model.resume, map_location="cpu", weights_only=False
        )
    msg = model.load_state_dict(checkpoint["model"], strict=True)
    logger.info(msg)
    required_keys = ["optimizer", "lr_scheduler", "epoch"]
    if all(key in checkpoint for key in required_keys):
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        epoch = checkpoint["epoch"] + 1
        logger.info(
            f"=> loaded successfully '{config.model.resume}' (epoch {checkpoint['epoch'] + 1})"
        )

    return epoch


def save_checkpoint(config, epoch, model, optimizer, lr_scheduler, logger):
    save_state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "epoch": epoch,
    }

    output_dir = os.path.join(config.train.output, config.model.name)
    save_path = os.path.join(output_dir, "checkpoints", f"ckpt_epoch_{epoch+1}.pt")
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")


def auto_resume_helper(output_dir, logger):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith("pt")]
    if len(checkpoints) > 0:
        logger.info(f"All checkpoints founded in {output_dir}: {checkpoints}")
        latest_checkpoint = max(
            [os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime
        )
        logger.info(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def find_ckptpath(config):
    ckpt_path = config.OUTPUT
    ckpt_name = config.CKPT
    total_path = os.path.join(ckpt_path, ckpt_name)
    best_ckpt_path = os.path.join(ckpt_path, "best", ckpt_name)
    if os.path.exists(total_path):
        return total_path
    elif os.path.exists(best_ckpt_path):
        return best_ckpt_path
    elif os.path.exists(ckpt_name):
        return ckpt_name
    else:
        raise ValueError("Please check your path.")
