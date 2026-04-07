import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import json
import time
import random
from LIB_model import ResNetBBoxModel
import sys
# Import dataloader
from LIB_dataloader import DatasetSDXL

# Note: `num_classes` is currently set explicitly in this script.
# Update it if training on a dataset with a different number of categories.


def parse_args():
    """
    Parse command-line arguments for hyperparameter configuration.
    """
    parser = argparse.ArgumentParser(description="Train ResNet for classification + 4-coordinate bbox regression (optional heads).")

    # Job / Logging
    parser.add_argument("--job_id", type=str, default="0000",
                        help="A name/ID for this training run (used in checkpoint filenames & subfolder).")

    # Data
    parser.add_argument("--train_data_dir", type=str,
                        default="path/to/SDXL_dataset_train",
                        help="Path to the training dataset directory. Replace this default with the path to your local training split; the directory must contain 'labels.h5'.")
    parser.add_argument("--val_data_dir", type=str,
                        default="/n/netscratch/sompolinsky_lab/Lab/ltiberi/datasets/project_manifolds/gen_ai_dataset/SDXL_dataset_validation",
                        help="Path to the validation dataset directory (must contain 'labels.h5').")

    # NEW: category holdout (exclude a deterministic random subset from training)
    parser.add_argument("--exclude_categories_Q", type=int, default=0,
                        help="If >0, exclude Q randomly-chosen categories from the TRAIN dataset "
                             "(deterministic via --exclude_categories_seed).")
    parser.add_argument("--exclude_categories_seed", type=int, default=1,
                        help="Seed for choosing which categories to exclude when --exclude_categories_Q > 0.")
    parser.add_argument("--exclude_categories_in_val", action="store_true",
                        help="If set, apply the same category exclusion to the validation dataset as well.")


    # Model
    parser.add_argument("--resnet_version", type=str, default="resnet50",
                        choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
                        help="Which ResNet variant to use.")
    parser.add_argument(
        "--active_heads",
        nargs=5,
        type=lambda x: x.lower() == 'true',
        default=[True, True, True, True, True],
        help="Specify 5 booleans (True/False) for active heads, e.g. --active_heads True False False True True. "
             "Order of heads is classification, center_x, center_y, width_x, width_y"
    )
    parser.add_argument("--per_class_bbox", action="store_true",
                        help="If set, each bbox coord predicts a value per class (output dim = num_classes).")

    # Freezing schedule
    parser.add_argument(
        "--freeze_schedule",
        nargs="*",
        default=["1:layer4"],
        help=(
            "A list of epoch:layer pairs, e.g. '--freeze_schedule 1:layer1 5:layer2', "
            "meaning freeze the backbone up to 'layer1' included at epoch 1, "
            "then freeze up to 'layer2' at epoch 5, etc. "
            "If not provided, no scheduled changes occur. "
            "Valid layers for ResNet: conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4."
            "option to unfreeze the full backbone: unfreeze"
        )
    )

    # Training Hyperparameters
    parser.add_argument("--batch_size", type=int, default=256, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate; if omitted an optimiser-specific default is used.")
    parser.add_argument("--weight_decay", type=float, default=None,
                        help="Weight decay; optimiser-specific default if omitted.")
    parser.add_argument("--momentum", type=float, default=None,
                        help="Momentum (SGD only). Ignored by Adam/AdamW.")
    parser.add_argument("--optimizer", type=str, default="AdamW",
                        choices=["SGD", "Adam", "AdamW"],
                        help="Which optimiser to use.")
    parser.add_argument("--regression_loss", type=str, default="SmoothL1",
                        choices=["SmoothL1", "L1", "MSE"],
                        help="Loss for each bbox coordinate.")

    # Loss Weights
    parser.add_argument("--cls_weight", type=float, default=1.0,
                        help="Weight for classification loss. If the model has no class_head, this is ignored.")
    parser.add_argument("--bbox_coord_weights", nargs=4, type=float, default=[1.0, 1.0, 1.0, 1.0],
                        help="List of 4 floats for weighting each of the 4 bbox-coordinates individually.")

    # Other
    parser.add_argument("--num_workers", type=int, default=16,
                        help="Number of Dataloader workers")
    parser.add_argument("--save_dir", type=str, default="./results",
                        help="Directory where checkpoints will be saved. "
                             "A subfolder named job_id will be created under this path.")
    parser.add_argument("--save_every", type=int, default=1,
                        help="Save checkpoint every X epochs.")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="Device to use for training (if you have a GPU, use 'cuda').")
    parser.add_argument("--time_per_batch", action="store_true",
                        help="If set, will print how much time each training batch takes.")
    parser.add_argument("--time_per_epoch", action="store_true",
                        help="If set, will print how much time each epoch takes.")

    return parser.parse_args()


def parse_freeze_schedule(schedule_list):
    """
    Given a list of strings like ["1:layer1", "5:layer2"], parse them into
    a dict: {1: "layer1", 5: "layer2"} so we can freeze at the start of epoch 1, etc.
    """
    freeze_dict = {}
    for item in schedule_list:
        parts = item.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid format for freeze_schedule item '{item}'. Expected 'epoch:layer'.")
        epoch_str, layer = parts
        epoch_num = int(epoch_str)
        freeze_dict[epoch_num] = layer
    return freeze_dict


def maybe_resume_checkpoint(subfolder, run_name, device):
    latest = os.path.join(subfolder, f"{run_name}_latest.pth")
    if not os.path.isfile(latest):
        return None, None, 0, None, {}

    ckpt = torch.load(latest, map_location="cpu", weights_only=False)        # ① load on CPU
    model = ckpt["model"].to(device)
    return (model,
            ckpt["optimizer_state_dict"],
            ckpt["epoch"],
            ckpt.get("transform"),
            ckpt.get("hyperparams", {}))


def format_mse_values(mse_sums, counts):
    """
    Convert each coordinate's aggregated MSE to a string.
    If the model's head for that coordinate was never used (count=0), print N/A.
    Otherwise, print the average MSE as x.xxx.
    """
    out = []
    for i, val in enumerate(mse_sums):
        if counts[i] == 0:
            out.append("N/A")
        else:
            avg = val / counts[i]
            out.append(f"{avg:.4f}")
    return f"({', '.join(out)})"


def format_float_or_na(value, count):
    """
    If 'count' is 0 (meaning the head wasn't active at all) or value is None, return 'N/A',
    else return formatted float with 4 decimals.
    """
    if value is None or count == 0:
        return "N/A"
    return f"{value:.4f}"


def main():
    args = parse_args()
    heads_bools = args.active_heads

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    freeze_dict = parse_freeze_schedule(args.freeze_schedule)

    # Build a run_name prefix
    heads_str = ''.join(['T' if b else 'F' for b in heads_bools])
    pc_tag = "PerClassReg_" if args.per_class_bbox else ""
    run_name = (f"{args.job_id}_"
                f"{args.resnet_version}_"
                f"heads{heads_str}_"
                f"{pc_tag}"
                f"LossW_{args.cls_weight}_{args.bbox_coord_weights[0]}_{args.bbox_coord_weights[1]}_{args.bbox_coord_weights[2]}_{args.bbox_coord_weights[3]}_")

    # Make subfolder for this job_id
    results_subfolder = os.path.join(args.save_dir, args.job_id)
    os.makedirs(results_subfolder, exist_ok=True)

    # RESUME INFO FROM CHECKPOINT IF IT EXISTS
    model, opt_state, start_epoch, saved_transform, saved_hparams = maybe_resume_checkpoint(
        results_subfolder, run_name, device
    )

    # DEFINE TRANSFORM - WILL BE OVERRIDEN IF WE RESUM
    default_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Must match the input_resolution of the model
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    transform_for_dataloaders = saved_transform or default_transform

    # ----------------------------
    # Prepare DataLoaders
    # ----------------------------

    # NEW: exclude a deterministic random subset of categories from training (and optionally validation)
    excluded_categories = []
    if args.exclude_categories_Q is not None and args.exclude_categories_Q > 0:
        # Build the category list deterministically from the on-disk folder structure.
        # (This matches the "category prefix" notion used by DatasetSDXL: image_keys like "cat/img_00123".)
        all_categories_for_exclusion = sorted([
            d for d in os.listdir(args.train_data_dir)
            if os.path.isdir(os.path.join(args.train_data_dir, d)) and d.lower() != "crashes"
        ])

        if args.exclude_categories_Q >= len(all_categories_for_exclusion):
            raise ValueError(
                f"--exclude_categories_Q={args.exclude_categories_Q} must be < number of categories "
                f"({len(all_categories_for_exclusion)}) found in {args.train_data_dir}"
            )

        rng = random.Random(args.exclude_categories_seed)
        excluded_categories = rng.sample(all_categories_for_exclusion, args.exclude_categories_Q)
        excluded_categories.sort()
        print(f"[Category holdout] Excluding {len(excluded_categories)} categories from TRAIN: {excluded_categories}",
              flush=True)

    # Store for logging/checkpoint hyperparams
    args.excluded_categories = excluded_categories

    train_dataset = DatasetSDXL(
        dataset_folder=args.train_data_dir,
        transform=transform_for_dataloaders,
        exclude_categories=excluded_categories if excluded_categories else None,
    )
    val_dataset = DatasetSDXL(
        dataset_folder=args.val_data_dir,
        transform=transform_for_dataloaders,
        exclude_categories=(excluded_categories if (args.exclude_categories_in_val and excluded_categories) else None),
    )

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=True)

    # ----------------------------
    #  CREATE MODEL (IF NOT RETRIEVED FROM CHECKPOINT)
    # ----------------------------
    if model is None:
        heads_bools = args.active_heads
        model = ResNetBBoxModel(
            resnet_version=args.resnet_version,
            num_classes=265,
            bbox_heads=heads_bools[1:5],
            class_head=heads_bools[0],
            per_class_bbox=args.per_class_bbox,
        ).to(device)
        start_epoch = 0

    # ----------------------------
    #  3. Define Losses & Optim
    # ----------------------------

    # ------------ regression loss -------------
    reg_loss_name = saved_hparams.get("regression_loss", args.regression_loss)

    if reg_loss_name == "SmoothL1":
        reg_criterion = nn.SmoothL1Loss(reduction="mean")
    elif reg_loss_name == "L1":
        reg_criterion = nn.L1Loss(reduction="mean")
    elif reg_loss_name == "MSE":
        reg_criterion = nn.MSELoss(reduction="mean")
    else:
        raise ValueError(f"Unknown regression_loss {reg_loss_name}")

    # ------------ classification loss -------------
    cls_criterion = nn.CrossEntropyLoss()

    # ------------ optimiser --------------------
    # ------------ optimiser --------------------
    opt_name = saved_hparams.get("optimizer", args.optimizer)

    # ---- fill optimiser-specific defaults ----
    if args.lr is None:
        lr = 0.02 * args.batch_size / 256 if opt_name == "SGD" else 1e-4
    else:
        lr = args.lr

    if args.weight_decay is None:
        weight_decay = 1e-4 if opt_name in ["SGD", "AdamW"] else 0.0
    else:
        weight_decay = args.weight_decay

    if opt_name == "SGD":
        momentum = args.momentum if args.momentum is not None else 0.9

    # ---- create optimiser ----
    if opt_name == "SGD":
        optimizer = optim.SGD(model.parameters(),
                              lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif opt_name == "Adam":
        optimizer = optim.Adam(model.parameters(),
                               lr=lr, weight_decay=weight_decay)
    elif opt_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(),
                                lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimiser {opt_name}")

    # ---- resume optimiser state if any ----
    if opt_state is not None:
        optimizer.load_state_dict(opt_state)

    # ---- cosine-annealing scheduler (SGD only) ----
    scheduler = None
    if opt_name == "SGD":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=0.0, last_epoch=start_epoch - 1)

    if opt_state is not None:
        optimizer.load_state_dict(opt_state)

    # Recover performance history or initialize anew
    perf_path = os.path.join(results_subfolder, f"{run_name}_performance_history.json")
    if os.path.isfile(perf_path):
        with open(perf_path) as f:
            perf_data = json.load(f)
        train_history = perf_data.get("train_performance", [])
        val_history = perf_data.get("val_performance", [])
        if train_history:
            # Ensure start_epoch matches the last entry in history
            last_logged_epoch = train_history[-1]["epoch"]
            start_epoch = max(start_epoch, last_logged_epoch)
        print(f"Resumed performance history up to epoch {start_epoch}.")
    else:
        train_history, val_history = [], []

    # For convenience, define a helper to dump performance after each checkpoint
    def save_performance_history():
        perf_filename = f"{run_name}_performance_history.json"
        perf_path = os.path.join(results_subfolder, perf_filename)
        performance_data = {
            "train_performance": train_history,
            "val_performance": val_history,
            "args": vars(args),
        }
        with open(perf_path, "w") as f:
            json.dump(performance_data, f, indent=2)
        print(f"Performance history saved: {perf_path}", flush=True)

    # ----------------------------
    #  4. Training/Validation Helpers
    # ----------------------------
    def compute_losses(images, categories, bboxes):
        """
        Forward pass, compute classification loss (if head is present)
        and each coordinate's loss separately (if head != None).
        Return (total_loss, cls_loss_val or None, coord_mses [float or None]).
        """
        class_logits, bbox_preds_list = model(images)

        # Classification loss
        cls_loss_val = None
        if class_logits is not None:
            cls_loss_val = cls_criterion(class_logits, categories)
        else:
            # If the classification head is absent, we keep cls_loss_val as None
            pass

        # We'll track the MSE for each coordinate for printing.
        # If a head is missing, store None.
        coord_mses = [None, None, None, None]

        # Weighted sum of the coordinate losses
        total_coord_loss = 0.0

        for i in range(4):
            head_output = bbox_preds_list[i]
            if head_output is not None:
                # Support either global head [B,1] or per-class head [B,C]
                if head_output.dim() == 2 and head_output.size(1) > 1:
                    # Per-class: select the column for the GT class
                    # categories: [B] (Long), gather expects [B,1]
                    pred_coord = head_output.gather(1, categories.view(-1, 1)).squeeze(1)  # [B]
                else:
                    # Global (current behavior)
                    pred_coord = head_output.squeeze(1)  # [B]
                true_coord = bboxes[:, i]

                c_loss = reg_criterion(pred_coord, true_coord)
                weighted_loss = args.bbox_coord_weights[i] * c_loss
                total_coord_loss += weighted_loss

                # For logging, let's always compute MSE, even if training with SmoothL1
                mse_criterion = nn.MSELoss()
                c_mse = mse_criterion(pred_coord, true_coord).item()
                coord_mses[i] = c_mse

        total_loss = 0.0
        if cls_loss_val is not None:
            total_loss += args.cls_weight * cls_loss_val
        total_loss += total_coord_loss

        return total_loss, cls_loss_val, coord_mses

    def compute_accuracy(logits, targets):
        """Compute classification accuracy if classification head is not None."""
        if logits is None:
            return None, 0  # No classification
        _, predicted = torch.max(logits, dim=1)
        correct_cls = (predicted == targets).sum().item()
        total_samples = targets.size(0)
        return correct_cls, total_samples

    # ----------------------------
    #  5. Training Loop
    # ----------------------------
    best_val_loss = float("inf")

    for epoch in range(start_epoch + 1, args.epochs + 1):
        epoch_start_time = time.time()

        # Check freeze schedule
        if epoch in freeze_dict:
            layer_to_freeze = freeze_dict[epoch]
            if layer_to_freeze == "unfreeze":
                model.unfreeze_backbone()
                print(f"==> UNfreezing full backbone at the start of epoch {epoch}", flush=True)
            else:
                print(f"==> Freezing backbone up to {layer_to_freeze} at the start of epoch {epoch}", flush=True)
                model.freeze_backbone_until(layer_to_freeze)

        # ---- TRAIN ----
        model.train()
        running_loss = 0.0
        running_cls_loss = 0.0  # Will accumulate numeric values only if classification head is present
        cls_count = 0          # Number of samples that had classification head

        running_coords_mse_sums = [0.0, 0.0, 0.0, 0.0]  # sum of MSEs for each coord
        coords_counts = [0, 0, 0, 0]                   # how many samples had each head
        correct_cls_accum = 0
        total_samples_cls = 0
        total_samples = 0

        for batch_idx, (images, (categories, bboxes)) in enumerate(train_loader):
            batch_start_time = time.time()

            images = images.to(device)
            categories = categories.to(device)
            bboxes = bboxes.to(device)

            optimizer.zero_grad()
            total_loss, cls_val, coord_mses = compute_losses(images, categories, bboxes)
            total_loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            running_loss += total_loss.item() * batch_size

            # Accumulate classification loss if available
            if cls_val is not None:
                running_cls_loss += cls_val.item() * batch_size
                cls_count += batch_size

            # Accumulate MSE sums
            for i in range(4):
                if coord_mses[i] is not None:
                    running_coords_mse_sums[i] += coord_mses[i] * batch_size
                    coords_counts[i] += batch_size

            # classification accuracy
            with torch.no_grad():
                class_logits, _ = model(images)
                correct_cls, samples_cls = compute_accuracy(class_logits, categories)
                if correct_cls is not None:
                    correct_cls_accum += correct_cls
                    total_samples_cls += samples_cls

            total_samples += batch_size

            if args.time_per_batch:
                elapsed = time.time() - batch_start_time
                print(
                    f"[Batch Timing] Epoch {epoch}, Batch {batch_idx + 1}/{len(train_loader)} took {elapsed:.3f} sec.", flush=True)

        epoch_loss = running_loss / total_samples

        # If classification head was never used, store None. Otherwise, compute average
        epoch_cls_loss = (running_cls_loss / cls_count) if cls_count > 0 else None
        epoch_acc = (correct_cls_accum / total_samples_cls) if total_samples_cls > 0 else None
        train_mse_str = format_mse_values(running_coords_mse_sums, coords_counts)

        # Print training summary
        msg_cls_loss = format_float_or_na(epoch_cls_loss, cls_count)
        msg_acc = format_float_or_na(epoch_acc, cls_count)
        print(f"[Train] Epoch {epoch} | "
              f"Loss: {epoch_loss:.4f} | "
              f"ClsLoss: {msg_cls_loss} | "
              f"MSE coords: {train_mse_str} | "
              f"Acc: {msg_acc}", flush=True)

        # Save training performance for plotting
        train_history.append({
            "epoch": epoch,
            "loss": epoch_loss,
            "cls_loss": epoch_cls_loss,
            "coord_mse": [
                (running_coords_mse_sums[i] / coords_counts[i]) if coords_counts[i] > 0 else None
                for i in range(4)
            ],
            "acc": epoch_acc,
        })

        # ---- VALIDATION ----
        model.eval()
        val_running_loss = 0.0
        val_running_cls_loss = 0.0
        val_cls_count = 0

        val_running_coords_mse_sums = [0.0, 0.0, 0.0, 0.0]
        val_coords_counts = [0, 0, 0, 0]
        val_correct_cls_accum = 0
        val_total_samples_cls = 0
        val_total_samples = 0

        with torch.no_grad():
            for images, (categories, bboxes) in val_loader:
                images = images.to(device)
                categories = categories.to(device)
                bboxes = bboxes.to(device)

                total_loss, cls_val, coord_mses = compute_losses(images, categories, bboxes)

                batch_size = images.size(0)
                val_running_loss += total_loss.item() * batch_size

                if cls_val is not None:
                    val_running_cls_loss += cls_val.item() * batch_size
                    val_cls_count += batch_size

                for i in range(4):
                    if coord_mses[i] is not None:
                        val_running_coords_mse_sums[i] += coord_mses[i] * batch_size
                        val_coords_counts[i] += batch_size

                class_logits, _ = model(images)
                correct_cls, samples_cls = compute_accuracy(class_logits, categories)
                if correct_cls is not None:
                    val_correct_cls_accum += correct_cls
                    val_total_samples_cls += samples_cls

                val_total_samples += batch_size

        val_epoch_loss = val_running_loss / val_total_samples
        val_epoch_cls_loss = (val_running_cls_loss / val_cls_count) if val_cls_count > 0 else None
        val_epoch_acc = (val_correct_cls_accum / val_total_samples_cls) if val_total_samples_cls > 0 else None
        val_mse_str = format_mse_values(val_running_coords_mse_sums, val_coords_counts)

        msg_val_cls_loss = format_float_or_na(val_epoch_cls_loss, val_cls_count)
        msg_val_acc = format_float_or_na(val_epoch_acc, val_cls_count)
        print(f"[Val]   Epoch {epoch} | "
              f"Loss: {val_epoch_loss:.4f} | "
              f"ClsLoss: {msg_val_cls_loss} | "
              f"MSE coords: {val_mse_str} | "
              f"Acc: {msg_val_acc}", flush=True)

        # Save validation performance for plotting
        val_history.append({
            "epoch": epoch,
            "loss": val_epoch_loss,
            "cls_loss": val_epoch_cls_loss,
            "coord_mse": [
                (val_running_coords_mse_sums[i] / val_coords_counts[i]) if val_coords_counts[i] > 0 else None
                for i in range(4)
            ],
            "acc": val_epoch_acc,
        })

        # ---- advance LR scheduler (SGD) ----
        if scheduler is not None:
            scheduler.step()

        # ---------------------------
        # Save checkpoint
        # ---------------------------
        if epoch % args.save_every == 0:
            ckpt_name = f"{run_name}_epoch{epoch}.pth"
            ckpt_path = os.path.join(results_subfolder, ckpt_name)

            checkpoint = {
                "epoch": epoch,  # or args.epochs for the final save
                "model": model,  # full object – NO state_dict
                "optimizer_state_dict": optimizer.state_dict(),
                "hyperparams": vars(args),
                "train_perf": train_history[-1],  # metrics of this epoch
                "val_perf": val_history[-1],
                "transform": transform_for_dataloaders,
            }

            torch.save(checkpoint, ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}", flush=True)

            # Always save a 'latest' pointer
            latest_path = os.path.join(results_subfolder, f"{run_name}_latest.pth")
            torch.save(checkpoint, latest_path)

            # Update best if needed
            if val_epoch_loss < best_val_loss:
                best_val_loss = val_epoch_loss
                best_path = os.path.join(results_subfolder, f"{run_name}_BEST.pth")
                torch.save(checkpoint, best_path)
                print(f"Best checkpoint updated (val_loss={val_epoch_loss:.4f}). Saved: {best_path}", flush=True)

            # Also save/update the performance history in a separate file
            save_performance_history()

        if args.time_per_epoch:
            epoch_elapsed = time.time() - epoch_start_time
            print(f"[Epoch Timing] Epoch {epoch} took {epoch_elapsed/60:.3f} min total.", flush=True)

    # Final save
    final_ckpt_path = os.path.join(results_subfolder, f"{run_name}_final.pth")
    final_ckpt = {
        "epoch": args.epochs,  # or args.epochs for the final save
        "model": model,  # full object – NO state_dict
        "optimizer_state_dict": optimizer.state_dict(),
        "hyperparams": vars(args),
        "train_perf": train_history[-1],  # metrics of this epoch
        "val_perf": val_history[-1],
        "transform": transform_for_dataloaders,
    }
    torch.save(final_ckpt, final_ckpt_path)
    print(f"Final model saved: {final_ckpt_path}", flush=True)

    # Save final performance history as well
    save_performance_history()


if __name__ == "__main__":
    main()



