# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%

import os
import json
import random
from tqdm import tqdm
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    GenerationConfig,
)
import torch
import torch.nn.functional as F
from fancy_einsum import einsum
import einops

from record_utils import record_activations, get_module, untuple_tensor
from HookedQwen import convert_to_hooked_model
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--gen_length", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--slice_idx", type=int, default=0)
parser.add_argument("--slice_size", type=int, default=64)
parser.add_argument("--num_heads", type=int, default=0)
parser.add_argument("--num_mlp_vecs", type=int, default=200)
parser.add_argument("--prev_token_thresh", type=float, default=0.05)
parser.add_argument("--interv_type", type=str, default="previous_token_heads")


args = parser.parse_args()
seed = args.seed
gen_length = args.gen_length
batch_size = args.batch_size
assert batch_size > 1
slice_idx = args.slice_idx
slice_size = args.slice_size
num_heads = args.num_heads
num_mlp_vecs = args.num_mlp_vecs
prev_token_thresh = args.prev_token_thresh
interv_type = args.interv_type
assert interv_type in ["previous_token_heads", "random_heads", "glu"]


# %%

cos = F.cosine_similarity

# %%

base_dir = "[INSERT BASE DIRECTORY]"
output_dir = os.path.join(
    base_dir,
    f"generations/interv_{interv_type}_seed_{seed}_heads_{num_heads}_mlps_{num_mlp_vecs}_thresh_{prev_token_thresh}",
)
os.makedirs(output_dir, exist_ok=True)

# %%


def seed_all(seed, deterministic_algos=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    if deterministic_algos:
        torch.use_deterministic_algorithms()


def unembed(vector, lm_head, k=10):
    dots = einsum("vocab d_model, d_model -> vocab", lm_head, vector.to(lm_head.device))
    top_k = dots.topk(k).indices
    return top_k


def unembed_text(vector, lm_head, tokenizer, k=10):
    top_k = unembed(vector, lm_head, k=k)
    return tokenizer.batch_decode(top_k, skip_special_tokens=True)


# %%


def _add_o_proj_hook(model, layer_idx, head_idx):
    def hook(module, input, output):
        # output.shape: [batch, heads, seq, head_dim]
        output[:, :, head_idx, :] = 0
        return output

    module = model.model.layers[layer_idx].self_attn.hook_attn_out_per_head
    return module.register_forward_hook(hook)


# %%


def _turn_off_mlp(model, layer_idx, mlp_idxs):
    def hook(module, input, output):
        output[:, -1, mlp_idxs] = 0
        return output

    module = model.model.layers[layer_idx].mlp.hook_mlp_mid
    return module.register_forward_hook(hook)


# %%


def add_hooks(model, hook_config):
    handles = []
    for hook_module, layer, head_idx in hook_config:
        if hook_module == "attn_out":
            hook_func = _add_o_proj_hook
        elif hook_module == "mlp":
            hook_func = _turn_off_mlp
        handles.append(hook_func(model, layer, head_idx))
    return handles


# %%


@torch.no_grad()
def generate_hooked(
    model,
    input_ids,
    attention_mask,
    max_new_tokens,
    block_size,
    tokenizer,
    hook_config,
):
    """
    Generate text using a transformer language model with greedy sampling.

    Args:
        model: The auto-regressive transformer model that outputs logits.
        input_ids: A tensor of shape (batch_size, sequence_length) representing the initial token indices.
        max_new_tokens: The number of new tokens to generate.
        block_size: The maximum sequence length (context window) the model can handle.
        device: The device on which computations are performed.

    Returns:
        A tensor containing the original context concatenated with the generated tokens.
    """
    remove_all_hooks(model)
    device = model.device
    model.eval()  # Set the model to evaluation mode
    eos_token_id = tokenizer.eos_token_id

    input_ids = input_ids.clone()
    attention_mask = attention_mask
    batch_size = input_ids.shape[0]

    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    token_open = tokenizer.encode(" (")[0]  # 320
    token_newline = tokenizer.encode("\n", add_special_tokens=False)[0]
    token_double_newline = tokenizer.encode("\n\n", add_special_tokens=False)[0]
    token_bracket_newline = tokenizer.encode(">\n\n", add_special_tokens=False)[0]

    for _ in range(max_new_tokens):
        if finished.all():
            break

        if input_ids.shape[1] > block_size:
            idx_cond = input_ids[:, -block_size:]
            attn_mask_cond = attention_mask[:, -block_size:]
        else:
            idx_cond = input_ids
            attn_mask_cond = attention_mask

        position_ids = attn_mask_cond.long().cumsum(-1) - 1
        position_ids.masked_fill_(attn_mask_cond == 0, 1)

        output = model(
            idx_cond,
            attention_mask=attn_mask_cond,
            position_ids=position_ids,
            return_dict=True,
        )
        logits = output["logits"]
        logits = logits[:, -1, :]  # shape: (batch, vocab_size)
        next_token = torch.argmax(logits, dim=-1, keepdim=True)  # shape: (batch, 1)
        prev_token = idx_cond[:, -1].unsqueeze(1)

        most_recent_token = [
            tokenizer.decode(idx_cond[batch_idx, -1]) for batch_idx in range(batch_size)
        ]

        interv_batch_idx = []
        for batch_idx in range(batch_size):
            if most_recent_token[batch_idx] == " (":
                interv_batch_idx.append(batch_idx)

        if len(interv_batch_idx) > 0:

            handles = add_hooks(model, hook_config)
            interv_output = model(
                idx_cond[interv_batch_idx],
                attention_mask=attn_mask_cond[interv_batch_idx],
                position_ids=position_ids[interv_batch_idx],
                return_dict=True,
            )
            logits = interv_output["logits"]
            logits = logits[:, -1, :]  # shape: (batch, vocab_size)
            interv_next_token = torch.argmax(logits, dim=-1, keepdim=True)
            next_token[interv_batch_idx] = interv_next_token

            for handle in handles:
                handle.remove()

        reached_eos = next_token.squeeze(1) == eos_token_id
        reached_two_newline = (prev_token.squeeze(1) == token_newline) & (
            next_token.squeeze(1) == token_newline
        )
        reached_double_newline = next_token.squeeze(1) == token_double_newline
        reached_bracket_newline = next_token.squeeze(1) == token_bracket_newline
        new_finished = (~finished) & (
            reached_eos
            | reached_double_newline
            | reached_two_newline
            | reached_bracket_newline
        )

        finished |= new_finished
        next_token[finished] = eos_token_id

        # Append the predicted token to the sequence
        input_ids = torch.cat([input_ids, next_token], dim=1)
        new_mask = torch.ones(
            (batch_size, 1), dtype=attention_mask.dtype, device=device
        )
        attention_mask = torch.cat([attention_mask, new_mask], dim=1)

    return input_ids


# %%


def get_mlp_value_vecs(model):
    mlp_value_vecs = [layer.mlp.down_proj.weight.cpu() for layer in model.model.layers]
    # [n_layers, d_mlp (11008), d_model (2048)]
    return torch.stack(mlp_value_vecs, dim=0)


# %%


def remove_all_hooks(model):
    for (
        name,
        module,
    ) in model.named_modules():  # Recursively iterates through submodules
        if hasattr(module, "_forward_hooks"):
            for handle_id in list(module._forward_hooks.keys()):
                module._forward_hooks.pop(handle_id)


# %%

config = {
    "probe_path": os.path.join(base_dir, "data_and_ckpts/probe.pt"),
    "batch_size": batch_size,
    "seed": seed,
}

# %%

seed_all(config["seed"])
assert torch.cuda.is_available()

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")


# %%

qwen = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    trust_remote_code=True,
    device_map="auto",
)

# %%

convert_to_hooked_model(qwen)

# %%

generation_config = GenerationConfig(do_sample=False)

# %%

token_this = tokenizer.encode("this")[0]  # 574
token_equals = tokenizer.encode("equals")[0]
token_open = tokenizer.encode(" (")[0]  # 320
token_not = tokenizer.encode("not")[0]  # 1921

# %%

samples = torch.load(os.path.join(base_dir, "data_and_ckpts/icl_base_run.pt"))

# %%

probe_model = torch.load(config["probe_path"]).detach().cuda()


# %%


def run(actor, samples, hook_config, batch_size, max_gen_length):
    all_generations = []

    test_size = len(samples)
    all_generations = []
    for batch_idx in tqdm(range(0, test_size, batch_size)):
        curr_batch = samples[batch_idx : batch_idx + batch_size]

        _this_timestep = [sample["this_timestep"] - 1 for sample in curr_batch]
        _input_ids = [
            curr_batch[_idx]["response"][: _this_timestep[_idx]]
            for _idx in range(len(curr_batch))
        ]
        max_length = max(seq.shape[0] for seq in _input_ids)
        padded_input_ids = []
        for seq in _input_ids:
            pad_length = max_length - seq.shape[0]
            padded = F.pad(seq, (pad_length, 0), value=tokenizer.pad_token_id)
            padded_input_ids.append(padded)
        input_ids = torch.stack(padded_input_ids, dim=0)
        attention_mask = input_ids != tokenizer.pad_token_id

        hooked_output = generate_hooked(
            actor,
            input_ids,
            attention_mask,
            max_gen_length,
            2000,
            tokenizer,
            hook_config,
        )
        hooked_output_text = tokenizer.batch_decode(
            hooked_output, skip_special_tokens=True
        )

        generated_ids = hooked_output[:, input_ids.shape[1] :]
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        all_generations.extend(generated_text)
    return all_generations


# %%


def build_mlp_hook_config(actor, probe_model, layers, k):

    labels = [0, 1]
    value_vecs = get_mlp_value_vecs(actor)
    hook_config = []
    top_cos_scores = {label: [] for label in labels}
    for label in labels:
        for target_probe_layer in range(18, 36):
            #target_probe_layer = int(model_layer / 48 * 36)
            target_probe = probe_model[target_probe_layer, :, label]
            model_layer = int(target_probe_layer / 36 * 48)

            for layer_idx in range(0, model_layer + 1):
                cos_scores = cos(
                    value_vecs[layer_idx].to(target_probe.device),
                    target_probe.unsqueeze(-1),
                    dim=0,
                )
                _topk = cos_scores.topk(k=k)
                _values = [x.item() for x in _topk.values]
                _idxs = [x.item() for x in _topk.indices]
                topk = list(
                    zip(
                        _values,
                        _idxs,
                        [target_probe_layer] * _topk.indices.shape[0],
                        [layer_idx] * _topk.indices.shape[0],
                    )
                )
                top_cos_scores[label].extend(topk)

    _sorted_scores_0 = sorted(top_cos_scores[0], key=lambda x: x[0], reverse=True)
    _sorted_scores_1 = sorted(top_cos_scores[1], key=lambda x: x[0], reverse=True)

    _unique = set()
    sorted_scores_0 = []
    for entry in _sorted_scores_0:
        _pair = (entry[3], entry[1])
        if _pair not in _unique:
            _unique.add(_pair)
            sorted_scores_0.append(("mlp", _pair[0], _pair[1]))

    _unique = set()
    sorted_scores_1 = []
    for entry in _sorted_scores_1:
        _pair = (entry[3], entry[1])
        if _pair not in _unique:
            _unique.add(_pair)
            sorted_scores_1.append(("mlp", _pair[0], _pair[1]))

    return sorted_scores_0, sorted_scores_1


# %%


def _get_occurrence_idxs(hay, needle):
    window_size = needle.shape[0]
    hay = hay.unfold(0, window_size, 1)
    mask = (hay == needle).all(dim=1)
    offset = window_size - 1
    match_idxs = mask.nonzero(as_tuple=True)[0] + offset
    return match_idxs


@torch.no_grad()
def _get_attn_density_for_target(actor, samples, batch_size):

    device = actor.device
    n_layers = 48
    record_module_names = [
        f"model.layers.{idx}.self_attn.hook_attn_pattern" for idx in range(n_layers)
    ]
    test_size = len(samples)
    _all_attn_pattern = []
    all_recording = {}
    cutoff = (
        tokenizer(" Let's try different", return_tensors="pt")["input_ids"]
        .squeeze()
        .to(device)
    )
    all_attn_density = []
    for batch_idx in tqdm(range(0, test_size, batch_size)):
        curr_batch = samples[batch_idx : batch_idx + batch_size]
        _this_timestep = [sample["this_timestep"] - 1 for sample in curr_batch]

        _input_ids = [
            curr_batch[_idx]["response"][: _this_timestep[_idx]]
            for _idx in range(len(curr_batch))
        ]
        max_length = max(seq.shape[0] for seq in _input_ids)
        padded_input_ids = []
        for seq in _input_ids:
            pad_length = max_length - seq.shape[0]
            padded = F.pad(seq, (pad_length, 0), value=tokenizer.pad_token_id)
            padded_input_ids.append(padded)
        input_ids = torch.stack(padded_input_ids, dim=0)
        attention_mask = input_ids != tokenizer.pad_token_id
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        with record_activations(actor, record_module_names) as recording:
            output = actor(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                return_dict=True,
            )

        # [layers, batch, heads, seq]
        _attn_pattern = torch.stack(
            [
                recording[f"model.layers.{layer_idx}.self_attn.hook_attn_pattern"][0][
                    :, :, -1, :
                ].to("cuda:1")
                for layer_idx in range(n_layers)
            ]
        )
        attn_density = []
        for _idx in range(len(curr_batch)):
            target_tokens = tokenizer(
                str(curr_batch[_idx]["target"]), return_tensors="pt"
            )["input_ids"].squeeze()
            if target_tokens[0] == 151646:  # (BOS):
                target_tokens = target_tokens[1:]

            _context = input_ids[_idx]
            cutoff_idx = _get_occurrence_idxs(_context, cutoff)
            # curr_context = _context[: cutoff_idx[0]]
            match_idxs = _get_occurrence_idxs(
                _context, target_tokens.to(_context.device)
            ).to(_attn_pattern.device)

            # [layers, heads]
            density = _attn_pattern[:, _idx, :, match_idxs].sum(dim=-1)
            attn_density.append(density)

    all_attn_density = torch.stack(attn_density, dim=0)
    return all_attn_density.mean(dim=0)


def _get_prev_token_heads(actor, samples, batch_size, thresh=0.1):

    attn_pattern = _get_attn_density_for_target(actor, samples, batch_size)
    top_values, top_idxs = torch.topk(attn_pattern.flatten(), 1000)
    top_idxs = np.array(np.unravel_index(top_idxs.cpu().numpy(), attn_pattern.shape)).T
    prev_token_heads = top_idxs[
        (top_values > thresh).nonzero().squeeze().cpu()
    ].squeeze()
    return prev_token_heads, attn_pattern


# %%


def get_WO_WV_OV(actor):

    n_layers = 48
    n_heads = actor.config.num_attention_heads
    n_kv_heads = actor.config.num_key_value_heads
    n_kv_groups = n_heads // n_kv_heads
    W_O = []
    W_V = []
    for idx in range(n_layers):

        _W_O = actor.model.layers[idx].self_attn.o_proj.weight
        _W_O = einops.rearrange(_W_O, "m (n h)->n h m", n=n_heads)
        W_O.append(_W_O.cpu())

        _W_V = actor.model.layers[idx].self_attn.v_proj.weight
        _W_V = einops.rearrange(_W_V, "(n h) m->n m h", n=n_kv_heads)
        _W_V = torch.repeat_interleave(_W_V, dim=0, repeats=n_kv_groups)
        W_V.append(_W_V.cpu())

    # [layers, heads, d_head, d_model]
    W_O = torch.stack(W_O, dim=0)
    W_V = torch.stack(W_V, dim=0)
    OV = einsum(
        "layers heads d_head d_model, layers heads d_model d_head -> layers heads d_model",
        W_O,
        W_V,
    )
    return W_O, W_V, OV


def get_OV_for_attn_heads(actor, OV, attn_heads):
    OVs = []
    for attn_head in attn_heads:
        layer_idx = attn_head[0]
        head_idx = attn_head[1]
        OVs.append(OV[layer_idx, head_idx].cpu())
    return torch.stack(OVs, dim=0)


def get_verification_heads(
    actor, samples, prev_token_heads, probe_model, num_mlp_vecs=200
):
    _, top_scores_1 = build_mlp_hook_config(actor, probe_model, list(range(24, 48)), 50)
    top_scores_1 = [(x[1], x[2]) for x in top_scores_1]

    gate_vecs = torch.stack(
        [
            actor.model.layers[x[0]].mlp.gate_proj.weight[x[1]].cpu()
            for x in top_scores_1[:num_mlp_vecs]
        ],
        dim=0,
    )
    up_proj_vecs = torch.stack(
        [
            actor.model.layers[x[0]].mlp.up_proj.weight[x[1]].cpu()
            for x in top_scores_1[:num_mlp_vecs]
        ],
        dim=0,
    )

    W_O, W_V, _OV = get_WO_WV_OV(actor)
    OV = get_OV_for_attn_heads(actor, _OV, prev_token_heads)

    gate_vecs = gate_vecs.to("cuda:1")
    up_proj_vecs = up_proj_vecs.to("cuda:1")
    OV = OV.to("cuda:1")

    dots_gate = einsum("N d_model, L d_model -> N L", gate_vecs, OV)
    act_fn = actor.model.layers[0].mlp.act_fn
    acts = act_fn(dots_gate)

    dots_up_proj = einsum("N d_model, L d_model -> N L", up_proj_vecs, OV)
    weights = (acts * dots_up_proj).mean(dim=0)
    top_val, top_idx = torch.topk(weights.flatten(), k=len(prev_token_heads))
    top_idx = np.array(
        np.unravel_index(top_idx.cpu().numpy(), weights.shape)
    ).T.squeeze()
    verif_heads = [prev_token_heads[x].tolist() for x in top_idx]
    return verif_heads


def build_attn_hook_config(
    actor,
    samples,
    batch_size,
    probe_model,
    prev_token_thresh=0.1,
    num_mlp_vecs=200,
):
    prev_token_heads, attn_pattern = _get_prev_token_heads(
        actor, samples, batch_size, prev_token_thresh
    )
    print(f"Number of prev token heads: {len(prev_token_heads)}")
    verif_heads = get_verification_heads(
        actor, samples, prev_token_heads, probe_model, num_mlp_vecs
    )
    heads = [("attn_out", layer, head_idx) for layer, head_idx in verif_heads]
    return heads, attn_pattern


# %%


def get_transfer_matrix(device):
    """
    Get transfer matrix between model `src` and model `dst`.
    """
    src_token_embeds = torch.load(
        os.path.join(base_dir, "data_and_ckpts/qwen2.5_3B_lm_head.pt")
    )
    num_samples = 100000

    # Grab 100000 random indices from the tokenizer
    random_tokens = torch.randint(0, src_token_embeds.shape[0], (num_samples,))
    # [100, 1024]
    src_tokens = src_token_embeds[random_tokens]

    # dst_token_embeds = dst.lm_head.weight.detach()
    dst_token_embeds = torch.load(
        os.path.join(base_dir, "data_and_ckpts/qwen2.5_14B_lm_head.pt")
    )
    # [100, 1024]
    dst_tokens = dst_token_embeds[random_tokens]

    # Random initialization
    A = src_tokens.to(device)
    B = dst_tokens.to(device)

    # Ensure that A and B have the same number of rows
    assert A.size(0) == B.size(0), "A and B must have the same number of rows."

    # Ridge parameter (regularization strength)
    alpha = 0.1

    # Setup the problem in terms of least squares with regularization
    A_transpose_A = A.t() @ A
    I = torch.eye(A.shape[1]).to(A_transpose_A.device)

    # Regularized least squares
    reg_matrix = A_transpose_A + alpha * I
    A_transpose_B = A.t() @ B

    # Compute T with regularization
    T = torch.linalg.solve(reg_matrix, A_transpose_B)

    # To verify the transformation
    B_transformed = A @ T
    error = torch.norm(B - B_transformed)
    return T, error


def get_transfer_probe(orig_probe):
    device = orig_probe.device
    transfer_matrix, error = get_transfer_matrix(device)
    return einsum(
        "layer d_model class, d_model d_model_hat -> layer d_model_hat class",
        orig_probe.to(device),
        transfer_matrix.to(device),
    )

# %%

# [Layer, d_model (Qwen), 2]
new_probe = get_transfer_probe(probe_model)


# %%

# MLP
if interv_type == "glu":

    print("Running MLP")
    mlp_hook_config_0, mlp_hook_config_1 = build_mlp_hook_config(
        qwen, new_probe, list(range(24, 48)), 50
    )
    num_mlps = 50 * 24
    mlp_both_hook_config = (
        mlp_hook_config_0[:num_mlps] + mlp_hook_config_1[:num_mlps]
    )
    generations = run(
        qwen,
        samples[slice_idx : slice_idx + slice_size],
        mlp_both_hook_config,
        batch_size,
        gen_length,
    )

elif interv_type == "random_heads":
    prev_heads, _ = build_attn_hook_config(
        qwen,
        samples,
        batch_size,
        new_probe,
        prev_token_thresh=prev_token_thresh,
        num_mlp_vecs=num_mlp_vecs,
    )
    if num_heads == 0:
        num_heads = len(prev_heads)

    hook_config = []
    for _ in range(num_heads):
        entry = ("attn_out", random.randint(0, 47), random.randint(0, 39))
        while entry in prev_heads or entry in hook_config:
            entry = ("attn_out", random.randint(0, 47), random.randint(0, 39))
        hook_config.append(entry)

    print(hook_config)
    generations = run(
        qwen,
        samples[slice_idx : slice_idx + slice_size],
        hook_config,
        batch_size,
        gen_length,
    )

elif interv_type == "previous_token_heads":
    print("Running Attention")
    hook_config, attn_pattern = build_attn_hook_config(
        qwen,
        samples,
        batch_size,
        new_probe,
        prev_token_thresh=prev_token_thresh,
        num_mlp_vecs=num_mlp_vecs,
    )

    if num_heads == 0:
        _hook_config = hook_config
    else:
        _hook_config = hook_config[:num_heads]

    generations = run(
        qwen,
        samples[slice_idx : slice_idx + slice_size],
        _hook_config,
        batch_size,
        gen_length,
    )


output_filepath = os.path.join(
    output_dir,
    f"slice{slice_idx}_size{slice_size}.json",
)
with open(output_filepath, "w") as f:
    json.dump(generations, f)
