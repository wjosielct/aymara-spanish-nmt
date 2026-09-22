# ============
# ./trainer.py
# ============

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


def run_training(config_path: str | Path = "config.yaml") -> None:

    start_time = time.perf_counter()

    cfg = load_config(config_path)

    # Deterministic environment and hardware configuration
    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda")
    print(f"Device: {device.type.upper()} | Mixed Precision (AMP): {use_amp} | Seed: {cfg['seed']}")

    tokenizer_dir = Path(cfg["tokenizer_dir"])
    tokenized_dir = Path(cfg["tokenized_dir"])
    models_dir = Path(cfg["models_dir"])
    results_dir = Path(cfg["results_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Tokenizer and Dataset Loading
    sp_model_path = tokenizer_dir / "SentencePiece.model"
    sp = spm.SentencePieceProcessor()
    sp.load(str(sp_model_path))
    pad_id, vocab_size = sp.pad_id(), sp.vocab_size()

    train_dataset = ParallelTranslationDataset(
        tsv_path=tokenized_dir / "train_bible_tokenized.tsv",
        src_col=f"{cfg['src']}_ids",
        tgt_col=f"{cfg['tgt']}_ids"
    )
    val_dataset = ParallelTranslationDataset(
        tsv_path=tokenized_dir / "val_bible_tokenized.tsv",
        src_col=f"{cfg['src']}_ids",
        tgt_col=f"{cfg['tgt']}_ids"
    )

    # In-Domain Test Set (Biblical Test Set)
    test_bible_dataset = ParallelTranslationDataset(
        tsv_path=tokenized_dir / "test_bible_tokenized.tsv",
        src_col=f"{cfg['src']}_ids",
        tgt_col=f"{cfg['tgt']}_ids"
    )

    # Out-of-Domain Test Set (Conversational Test Set)
    test_conv_dataset = ParallelTranslationDataset(
        tsv_path=tokenized_dir / "test_dialogue_tokenized.tsv",
        src_col=f"{cfg['src']}_ids",
        tgt_col=f"{cfg['tgt']}_ids"
    )

    batch_size = cfg["batch_size_cuda"] if device.type == "cuda" else cfg["batch_size_cpu"]

    g_train = torch.Generator()
    g_train.manual_seed(cfg["seed"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=g_train,
        worker_init_fn=seed_worker,
        collate_fn=lambda btch: collate_fn_pad(btch, pad_id)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda btch: collate_fn_pad(btch, pad_id)
    )

    # Transformer Model Initialization
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

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    # Optimizer and Scheduler
    epochs = cfg["epochs"]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=cfg["label_smoothing"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["lr"]),
        betas=(cfg["adam_beta1"], cfg["adam_beta2"]),
        eps=float(cfg["adam_eps"]),
        weight_decay=float(cfg["weight_decay"])
    )
    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=float(cfg["lr"]),
        total_steps=total_steps,
        pct_start=cfg["lr_pct_start"],
        anneal_strategy=cfg["anneal_strategy"]
    )
    scaler = GradScaler(enabled=use_amp)

    # Training and Validation Loop
    best_val_loss = float("inf")
    best_val_bleu = 0.0
    best_val_chrf = 0.0
    best_epoch = None
    history = []

    model_tag = f"{cfg['src']}_to_{cfg['tgt']}_Enc{cfg['encoder_layers']}_Dec{cfg['decoder_layers']}_Seed{cfg['seed']}"
    best_model_path = models_dir / f"trained_model_{model_tag}.pt"

    print("\n" + "=" * 70)
    print(f"TRAINING [{cfg['src'].upper()} --> {cfg['tgt'].upper()} | Enc:{cfg['encoder_layers']} Dec:{cfg['decoder_layers']}] -> {epochs} Epochs")
    print(f"Train Pairs: {len(train_dataset):,} | Val Pairs: {len(val_dataset):,}")
    print("=" * 70)

    for epoch in range(1, epochs + 1):
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

        # Validation of Loss
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

        # Periodic computation of BLEU / chrF++ on Validation Set
        val_bleu, val_chrf = None, None
        if epoch % cfg["val_eval_interval_epochs"] == 0 or epoch == epochs:
            val_bleu, val_chrf, _, _ = evaluate_metrics(
                model=model,
                dataset=val_dataset,
                sp=sp,
                device=device,
                beam_size=cfg["val_beam_size"],
                max_len=cfg["val_max_len"],
                alpha=cfg["val_alpha"]
            )

            if val_bleu > best_val_bleu:
                best_val_bleu = val_bleu

            if val_chrf > best_val_chrf:
                best_val_chrf = val_chrf
                best_epoch = epoch
                torch.save(model.state_dict(), best_model_path)

        bleu_str = f"{val_bleu:.2f}" if val_bleu is not None else "-"
        chrf_str = f"{val_chrf:.2f}" if val_chrf is not None else "-"
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val BLEU: {bleu_str} | Val chrF++: {chrf_str}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_bleu": val_bleu,
            "val_chrf": val_chrf
        })

        pd.DataFrame(history).to_csv(results_dir / f"training_history_{model_tag}.csv", index=False)

    print("\n" + "=" * 70)
    print(f"¡Training Complete! --> Best Epoch: {best_epoch} | Best Val chrF++: {best_val_chrf:.2f}")
    print("=" * 70)

    # =============================================================
    # Final Evaluation in Test Sets (Biblical vs Conversational)
    # =============================================================
    print(f"\nEvaluating Test Sets with --> {best_model_path}")

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

    eval_model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    eval_model.eval()

    # Evaluation in In-Domain Test Set (Biblical Test Set)
    bib_bleu, bib_chrf, bib_bleu_sig, bib_chrf_sig = evaluate_metrics(
        model=eval_model,
        dataset=test_bible_dataset,
        sp=sp,
        device=device,
        beam_size=cfg["test_beam_size"],
        max_len=cfg["test_max_len"],
        alpha=cfg["test_alpha"]
    )

    bib_results = {
        "test_pairs": len(test_bible_dataset),
        "bleu_score": bib_bleu,
        "chrf_score": bib_chrf,
        "bleu_signature": bib_bleu_sig,
        "chrf_signature": bib_chrf_sig
    }

    # Evaluation in Out-of-Domain Test Set (Conversational Test Set)
    conv_bleu, conv_chrf, conv_bleu_sig, conv_chrf_sig = evaluate_metrics(
        model=eval_model,
        dataset=test_conv_dataset,
        sp=sp,
        device=device,
        beam_size=cfg["test_beam_size"],
        max_len=cfg["test_max_len"],
        alpha=cfg["test_alpha"]
    )

    conv_results = {
        "test_pairs": len(test_conv_dataset),
        "bleu_score": conv_bleu,
        "chrf_score": conv_chrf,
        "bleu_signature": conv_bleu_sig,
        "chrf_signature": conv_chrf_sig
    }

    # Printing results on console
    print("\n" + "=" * 65)
    print(f"FINAL TEST SET EVALUATION (chrF++ Checkpoint | {model_tag})")
    print("=" * 65)

    print(f"• IN-DOMAIN TEST SET (Biblical Test Set) [{len(test_bible_dataset):,} pairs]")
    print(f"   - BLEU Score  : {bib_bleu:.2f}")
    print(f"   - chrF++ Score: {bib_chrf:.2f}")

    print(f"• OUT-OF-DOMAIN TEST SET (Conversational Test Set) [{len(test_conv_dataset):,} pairs]")
    print(f"   - BLEU Score  : {conv_bleu:.2f}")
    print(f"   - chrF++ Score: {conv_chrf:.2f}")
    print("=" * 65 + "\n")

    elapsed_time = time.perf_counter() - start_time

    # Exporting the consolidated JSON report
    eval_payload = {
        "model_tag": model_tag,
        "experiment_type": "Base-Training",
        "optimized_for": "chrF++",
        "best_checkpoint_epoch": best_epoch,
        "checkpoint": str(best_model_path),
        "biblical_results": bib_results,
        "conversational_results": conv_results,
        "total_execution_time": elapsed_time/3600,
        "config": cfg
    }

    out_json = f"trained_test_metrics_{model_tag}.json"

    with open(results_dir / out_json, "w", encoding="utf-8") as file:
        json.dump(eval_payload, file, indent=4)

    print(f"Total execution time: {elapsed_time / 3600:.2f} hours")
    print(f"[Saved] Results payload exported to --> {results_dir / out_json}")

