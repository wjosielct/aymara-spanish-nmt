# ./fine_tuner.py

import json
from pathlib import Path
import time
import warnings
import pandas as pd
import sentencepiece as spm
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from src.data.dataset import ParallelTranslationDataset, collate_fn_pad
from src.models.transformer import TranslationTransformer
from src.utils.metrics import evaluate_metrics
from src.utils.reproducibility import set_seed, seed_worker
from src.utils.config import load_config

warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.transformer")


def run_fine_tuning(config_path: str | Path = "config.yaml") -> None:

    cfg = load_config(config_path)

    # Deterministic environment and hardware configuration
    set_seed(cfg["seed"])
    start_time = time.perf_counter()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda")
    print(f"Device: {device.type.upper()} | Mixed Precision (AMP): {use_amp} | Seed: {cfg['seed']}")

    tokenizer_dir = Path(cfg["tokenizer_dir"])
    tokenized_dir = Path(cfg["tokenized_dir"])
    models_dir = Path(cfg["models_dir"])
    results_dir = Path(cfg["results_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Tokenizer Loading
    sp_model_path = tokenizer_dir / "SentencePiece.model"
    sp = spm.SentencePieceProcessor()
    sp.load(str(sp_model_path))
    pad_id, vocab_size = sp.pad_id(), sp.vocab_size()

    # Datasets for Fine-Tuning and Final Evaluation
    train_ft_dataset = ParallelTranslationDataset(
        tsv_path=tokenized_dir / "train_dialogue_tokenized.tsv",
        src_col=f"{cfg['src']}_ids",
        tgt_col=f"{cfg['tgt']}_ids"
    )
    val_ft_dataset = ParallelTranslationDataset(
        tsv_path=tokenized_dir / "val_dialogue_tokenized.tsv",
        src_col=f"{cfg['src']}_ids",
        tgt_col=f"{cfg['tgt']}_ids"
    )
    test_id_dataset = ParallelTranslationDataset(
        tsv_path=tokenized_dir / "test_bible_tokenized.tsv",
        src_col=f"{cfg['src']}_ids",
        tgt_col=f"{cfg['tgt']}_ids"
    )
    test_ood_dataset = ParallelTranslationDataset(
        tsv_path=tokenized_dir / "test_dialogue_tokenized.tsv",
        src_col=f"{cfg['src']}_ids",
        tgt_col=f"{cfg['tgt']}_ids"
    )

    g_train = torch.Generator()
    g_train.manual_seed(cfg["seed"])

    train_loader = DataLoader(
        train_ft_dataset,
        batch_size=cfg["ft_batch_size"],
        shuffle=True,
        generator=g_train,
        worker_init_fn=seed_worker,
        collate_fn=lambda btch: collate_fn_pad(btch, pad_id)
    )

    val_loader = DataLoader(
        val_ft_dataset,
        batch_size=cfg["ft_batch_size"],
        shuffle=False,
        collate_fn=lambda btch: collate_fn_pad(btch, pad_id)
    )

    # Load trained model with In-Domain dataset
    model_tag = f"{cfg['src']}_to_{cfg['tgt']}_Enc{cfg['encoder_layers']}_Dec{cfg['decoder_layers']}"
    base_checkpoint_path = models_dir / f"trained_model_{model_tag}.pt"

    if not base_checkpoint_path.exists():
        raise FileNotFoundError(
            f"[ERROR] Pre-trained base model not found at: {base_checkpoint_path}. "
            f"Please run 'train_and_evaluate.py' first."
        )

    print(f"\nLoading pre-trained base model --> {base_checkpoint_path}")
    model = TranslationTransformer(
        vocab_size=vocab_size,
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        num_encoder_layers=cfg["encoder_layers"],
        num_decoder_layers=cfg["decoder_layers"],
        dim_feedforward=cfg["dim_feedforward"],
        dropout=cfg["dropout"],
        pad_id=pad_id
    ).to(device)

    model.load_state_dict(torch.load(base_checkpoint_path, map_location=device, weights_only=True))

    # Optimizer and Scheduler (Reduced Learning Rate to preserve previous representations)
    ft_epochs = cfg["ft_epochs"]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=cfg["label_smoothing"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["ft_lr"]),
        betas=(cfg["adam_beta1"], cfg["adam_beta2"]),
        eps=float(cfg["adam_eps"]),
        weight_decay=float(cfg["weight_decay"])
    )
    total_steps = len(train_loader) * ft_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    scaler = GradScaler(enabled=use_amp)

    # Fine-Tuning Loop
    best_val_loss = float("inf")
    best_val_bleu = 0.0
    best_val_chrf = 0.0
    best_ft_model_path = models_dir / f"finetuned_model_{model_tag}.pt"
    history = []

    print("\n" + "=" * 75)
    print(f"OOD FINE-TUNING [{cfg['src'].upper()} --> {cfg['tgt'].upper()} | Enc:{cfg['encoder_layers']} Dec:{cfg['decoder_layers']}] -> {ft_epochs} Epochs")
    print(f"Train Pairs (OOD): {len(train_ft_dataset):,} | Val Pairs (OOD): {len(val_ft_dataset):,}")
    print("=" * 75)

    for epoch in range(1, ft_epochs + 1):
        model.train()
        train_loss = 0.0

        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:].contiguous()

            optimizer.zero_grad()
            with autocast(device_type=device.type, enabled=use_amp, dtype=torch.float16):
                logits = model(src, tgt_in)
                loss = criterion(logits.reshape(-1, vocab_size), tgt_out.reshape(-1))

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip_norm"]))
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip_norm"]))
                optimizer.step()

            scheduler.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation by Loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:].contiguous()
                with autocast(device_type=device.type, enabled=use_amp, dtype=torch.float16):
                    logits = model(src, tgt_in)
                    loss = criterion(logits.reshape(-1, vocab_size), tgt_out.reshape(-1))
                val_loss += loss.item()

        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        # Periodic validation by chrF++ / BLEU
        val_bleu, val_chrf = None, None
        if epoch % cfg["ft_val_eval_interval"] == 0 or epoch == ft_epochs:
            val_bleu, val_chrf, _, _ = evaluate_metrics(
                model=model,
                dataset=val_ft_dataset,
                sp=sp,
                device=device,
                beam_size=cfg["val_beam_size"],
                max_len=cfg["val_max_len"],
                alpha=cfg["val_alpha"],
                max_samples=None,
                seed=cfg["seed"]
            )

            if val_bleu > best_val_bleu:
                best_val_bleu = val_bleu

            if val_chrf > best_val_chrf:
                best_val_chrf = val_chrf
                torch.save(model.state_dict(), best_ft_model_path)

        bleu_str = f"{val_bleu:.2f}" if val_bleu is not None else "-"
        chrf_str = f"{val_chrf:.2f}" if val_chrf is not None else "-"
        print(f"Epoch {epoch:02d}/{ft_epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val BLEU: {bleu_str} | Val chrF++: {chrf_str}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_bleu": val_bleu,
            "val_chrf": val_chrf
        })

        pd.DataFrame(history).to_csv(results_dir / f"finetuning_history_{model_tag}.csv", index=False)

    print("\n" + "=" * 75)
    print(f"Fine-Tuning Complete! --> Best Val Loss: {best_val_loss:.4f} | Best Val BLEU: {best_val_bleu:.2f} | Best Val chrF++: {best_val_chrf:.2f}")
    print("=" * 75)

    # --------------------------------------------------------------
    # Final Evaluation in Test Sets (In-Domain vs Out-of-Domain)
    # --------------------------------------------------------------
    print(f"\nEvaluating Test Sets with --> {best_ft_model_path}")

    eval_model = TranslationTransformer(
        vocab_size=vocab_size,
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        num_encoder_layers=cfg["encoder_layers"],
        num_decoder_layers=cfg["decoder_layers"],
        dim_feedforward=cfg["dim_feedforward"],
        dropout=0.0,
        pad_id=pad_id
    ).to(device)

    eval_model.load_state_dict(torch.load(best_ft_model_path, map_location=device, weights_only=True))
    eval_model.eval()

    # Evaluation in In-Domain Test Set (Biblical)
    id_bleu, id_chrf, id_bleu_sig, id_chrf_sig = evaluate_metrics(
        model=eval_model,
        dataset=test_id_dataset,
        sp=sp,
        device=device,
        beam_size=cfg["test_beam_size"],
        max_len=cfg["test_max_len"],
        alpha=cfg["test_alpha"],
        max_samples=None,
        seed=cfg["seed"]
    )

    id_results = {
        "test_pairs": len(test_id_dataset),
        "bleu_score": id_bleu,
        "chrf_score": id_chrf,
        "sacrebleu_signature": id_bleu_sig,
        "chrf_signature": id_chrf_sig
    }

    # Evaluation in Out-of-Domain Test Set (Dialogues)
    ood_bleu, ood_chrf, ood_bleu_sig, ood_chrf_sig = evaluate_metrics(
        model=eval_model,
        dataset=test_ood_dataset,
        sp=sp,
        device=device,
        beam_size=cfg["test_beam_size"],
        max_len=cfg["test_max_len"],
        alpha=cfg["test_alpha"],
        max_samples=None,
        seed=cfg["seed"]
    )

    ood_results = {
        "test_pairs": len(test_ood_dataset),
        "bleu_score": ood_bleu,
        "chrf_score": ood_chrf,
        "sacrebleu_signature": ood_bleu_sig,
        "chrf_signature": ood_chrf_sig
    }

    # Printing results on console
    print("\n" + "=" * 65)
    print(f"FINAL TEST EVALUATION AFTER FINE-TUNING (chrF++ | {model_tag})")
    print("=" * 65)

    print(f"• IN-DOMAIN TEST (Biblical)  [{len(test_id_dataset):,} pairs]")
    print(f"   - BLEU Score  : {id_bleu:.2f}")
    print(f"   - chrF++ Score: {id_chrf:.2f}")

    print(f"• OUT-OF-DOMAIN TEST (Dialogue)  [{len(test_ood_dataset):,} pairs]")
    print(f"   - BLEU Score  : {ood_bleu:.2f}")
    print(f"   - chrF++ Score: {ood_chrf:.2f}")
    print("=" * 65 + "\n")

    # Exporting the consolidated JSON report
    eval_payload = {
        "model_tag": model_tag,
        "experiment_type": "Out-Of-Domain Fine-Tuning",
        "optimized_for": "chrF++",
        "base_checkpoint": str(base_checkpoint_path),
        "finetuned_checkpoint": str(best_ft_model_path),
        "in_domain_results": id_results,
        "out_of_domain_results": ood_results,
        "config": cfg
    }

    out_json = f"finetuned_test_metrics_{model_tag}.json"

    with open(results_dir / out_json, "w", encoding="utf-8") as file:
        json.dump(eval_payload, file, indent=4)

    elapsed_time = time.perf_counter() - start_time
    print(f"Fine-tuning total time: {elapsed_time / 60:.2f} minutes")
    print(f"[Saved] Results payload exported to --> {results_dir / out_json}")
