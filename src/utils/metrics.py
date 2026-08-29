# ./src/utils/metrics.py

import math
import random
from typing import List, Tuple
import sacrebleu
import sentencepiece as spm
import torch
import torch.nn.functional as F
from src.models.transformer import TranslationTransformer

@torch.no_grad()
def greedy_decode(
    model: TranslationTransformer,
    src: torch.Tensor,
    sp: spm.SentencePieceProcessor,
    max_len: int = 150,
    device: torch.device = torch.device("cpu")
) -> List[int]:

    """
    Fast greedy decoding (argmax).
    """

    model.eval()
    bos_id, eos_id, pad_id = sp.bos_id(), sp.eos_id(), sp.pad_id()

    src = src.unsqueeze(0).to(device)
    src_mask = (src == pad_id).to(device)

    src_emb = model.pos_encoder(model.embedding(src) * math.sqrt(model.d_model))
    memory = model.transformer.encoder(src_emb, src_key_padding_mask=src_mask)

    generated = [bos_id]
    
    for _ in range(max_len):
        tgt_tensor = torch.tensor([generated], dtype=torch.long, device=device)
        _, tgt_causal_mask = model.make_tgt_mask(tgt_tensor)
        tgt_emb = model.pos_encoder(model.embedding(tgt_tensor) * math.sqrt(model.d_model))

        out = model.transformer.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_causal_mask,
            memory_key_padding_mask=src_mask
        )
        logits = model.output_layer(out[:, -1, :])
        next_token = logits.argmax(dim=-1).item()
        generated.append(next_token)

        if next_token == eos_id:
            break

    return generated


@torch.no_grad()
def beam_search_decode(
    model: TranslationTransformer,
    src: torch.Tensor,
    sp: spm.SentencePieceProcessor,
    beam_size: int = 5,
    max_len: int = 150,
    alpha: float = 0.6,
    device: torch.device = torch.device("cpu")
) -> List[int]:
    
    """
    Beam search decoding with length normalization penalty (Wu et al., GNMT).
    """

    model.eval()
    bos_id, eos_id, pad_id = sp.bos_id(), sp.eos_id(), sp.pad_id()

    src = src.unsqueeze(0).to(device)
    src_mask = (src == pad_id).to(device)

    src_emb = model.pos_encoder(model.embedding(src) * math.sqrt(model.d_model))
    memory = model.transformer.encoder(src_emb, src_key_padding_mask=src_mask)

    # List of tuples: (sequence_ids, cumulative_log_prob)
    candidates = [([bos_id], 0.0)]
    completed_sequences = []

    for _ in range(max_len):
        new_candidates = []
        all_completed = True

        for seq, score in candidates:
            if seq[-1] == eos_id:
                # Calculate final score with length penalty
                lp = ((5.0 + len(seq)) / 6.0) ** alpha
                completed_sequences.append((seq, score / lp))
                continue

            all_completed = False
            tgt_tensor = torch.tensor([seq], dtype=torch.long, device=device)
            _, tgt_causal_mask = model.make_tgt_mask(tgt_tensor)
            tgt_emb = model.pos_encoder(model.embedding(tgt_tensor) * math.sqrt(model.d_model))

            out = model.transformer.decoder(
                tgt=tgt_emb,
                memory=memory,
                tgt_mask=tgt_causal_mask,
                memory_key_padding_mask=src_mask
            )

            # Log-probabilities for next token
            log_probs = F.log_softmax(model.output_layer(out[:, -1, :]), dim=-1)
            topk_log_probs, topk_ids = torch.topk(log_probs, beam_size, dim=-1)

            for k in range(beam_size):
                next_id = topk_ids[0, k].item()
                next_score = score + topk_log_probs[0, k].item()
                new_candidates.append((seq + [next_id], next_score))

        if all_completed:
            break

        # Keep only top beam_size hypotheses for next step
        candidates = sorted(new_candidates, key=lambda x: x[1], reverse=True)[:beam_size]

    # If no sequence emitted EOS, normalize remaining beams
    if not completed_sequences:
        for seq, score in candidates:
            lp = ((5.0 + len(seq)) / 6.0) ** alpha
            completed_sequences.append((seq, score / lp))

    # Return best hypothesis sequence
    best_sequence = sorted(completed_sequences, key=lambda x: x[1], reverse=True)[0][0]
    return best_sequence


@torch.no_grad()
def evaluate_metrics(
    model: TranslationTransformer,
    dataset,
    sp: spm.SentencePieceProcessor,
    device: torch.device,
    beam_size: int = 1,
    max_len: int = 150,
    alpha: float = 0.6,
    max_samples: int | None = None,
    seed: int = 1989,
) -> Tuple[float, float, str, str]:

    model.eval()
    total_samples = len(dataset)

    if max_samples and max_samples < total_samples:
        rng = random.Random(seed)
        indices = rng.sample(range(total_samples), max_samples)
    else:
        indices = range(total_samples)

    hypotheses = []
    references = []

    for idx in indices:
        src, tgt = dataset[idx]
        ref_text = sp.decode(tgt.tolist())

        if beam_size == 1:
            pred_ids = greedy_decode(model=model, src=src, sp=sp, max_len=max_len, device=device)
        else:
            pred_ids = beam_search_decode(
                model=model,
                src=src,
                sp=sp,
                beam_size=beam_size,
                max_len=max_len,
                alpha=alpha,
                device=device
            )

        pred_text = sp.decode(pred_ids)
        references.append(ref_text)
        hypotheses.append(pred_text)

    bleu_obj = sacrebleu.corpus_bleu(hypotheses, [references])
    chrf_obj = sacrebleu.corpus_chrf(hypotheses, [references], word_order=2)

    return bleu_obj.score, chrf_obj.score, str(bleu_obj), str(chrf_obj)