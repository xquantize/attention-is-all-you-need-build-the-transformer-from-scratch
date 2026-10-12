"""
Attention Is All You Need: Build the Transformer From Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - build_token_to_id_vocab
def build_token_to_id_vocab(sentences, specials=('<pad>', '<bos>', '<eos>', '<unk>')):
    # TODO: build a token-to-id dict with specials first, then corpus tokens in first-seen order.
    vocab = {token: idx for idx, token in enumerate(specials)}
    current_id = len(vocab)

    for sentence in sentences:
        tokens = sentence.split() if isinstance(sentence, str) else sentence

        for token in tokens:
            if token not in vocab:
                vocab[token] = current_id
                current_id += 1

    return vocab

# Step 2 - build_id_to_token_vocab
def build_id_to_token_vocab(token_to_id):
    return {idx: token for token, idx in token_to_id.items()}

# Step 3 - encode_sentence_to_ids
def encode_sentence_to_ids(sentence, token_to_id, unk_token='<unk>'):
    # TODO: convert whitespace tokens of `sentence` to ids via `token_to_id`, using `unk_token`'s id for OOV
    unk_id = token_to_id.get(unk_token)
    tokens = sentence.split() if isinstance(sentence, str) else sentence

    return [token_to_id.get(token, unk_id) for token in tokens]

# Step 4 - decode_ids_to_tokens
def decode_ids_to_tokens(ids, id_to_token):
    # TODO: map each id in ids to its token string via id_to_token and return the list
    return [id_to_token.get(idx, '<unk>') for idx in ids]

# Step 5 - pad_id_sequence
def pad_id_sequence(ids, max_len, pad_id):
    # TODO: return a list of length exactly max_len, padding with pad_id or truncating.
    if len(ids) > max_len:
        return list(ids[:max_len])

    return list(ids) + [pad_id] * (max_len - len(ids))

# Step 6 - stack_padded_sequences_to_batch
import torch

def stack_padded_sequences_to_batch(padded_sequences):
    """Stack a list of equal-length padded id sequences into a 2D LongTensor batch."""
    # TODO: stack padded id sequences into a (B, L) torch.long tensor
    return torch.tensor(padded_sequences, dtype=torch.long)

# Step 7 - scale_embeddings_by_sqrt_d_model
import math
import torch

def scale_embeddings_by_sqrt_d_model(embeddings, d_model):
    """Scale a token embedding tensor by sqrt(d_model)."""
    # TODO: rescale embeddings by sqrt(d_model) as in the original Transformer paper
    return embeddings * math.sqrt(d_model)

# Step 8 - compute_positional_div_term
import torch

def compute_positional_div_term(d_model):
    # TODO: return a 1D FloatTensor of length d_model // 2 holding the sinusoidal frequency divisors
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
    )

    return div_term

# Step 9 - build_position_index_column
import torch

def build_position_index_column(max_len):
    """Return a (max_len, 1) float tensor of [0, 1, ..., max_len-1]."""
    # TODO: build a column vector of position indices from 0 to max_len-1
    return torch.arange(max_len, dtype=torch.float32).unsqueeze(1)

# Step 10 - fill_even_indices_with_sin
import torch

def fill_even_indices_with_sin(pe, position, div_term):
    """Fill even feature indices of pe with sin(position * div_term)."""
    # TODO: write sin(position * div_term) into the even-indexed columns of pe and return it
    pe[:, 0::2] = torch.sin(position * div_term)

    return pe

# Step 11 - fill_odd_indices_with_cos
import torch

def fill_odd_indices_with_cos(pe, position, div_term):
    # TODO: fill the odd-indexed columns of pe with cos(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe

# Step 12 - build_sinusoidal_positional_encoding
import torch

def build_sinusoidal_positional_encoding(max_len, d_model):
    """Assemble the (max_len, d_model) sinusoidal positional encoding matrix."""
    # TODO: build the (max_len, d_model) sinusoidal positional encoding matrix
    pe = torch.zeros(max_len, d_model)

    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe

# Step 13 - add_positional_encoding_to_embeddings
import torch

def add_positional_encoding_to_embeddings(embedded_batch, positional_encoding):
    # TODO: add the first L rows of positional_encoding to embedded_batch and return the sum.
    seq_len = embedded_batch.size(1) if embedded_batch.ndim == 3 else embedded_batch.size(0)

    return embedded_batch + positional_encoding[:seq_len]

# Step 14 - build_padding_mask
import torch

def build_padding_mask(token_ids, pad_id):
    """Return a (B, 1, 1, L) bool mask: True where token_ids != pad_id."""
    # TODO: build a boolean mask marking non-pad positions, shaped for broadcasting against attention scores
    mask = (token_ids != pad_id)

    return mask.unsqueeze(1).unsqueeze(2)

# Step 15 - build_causal_mask
import torch

def build_causal_mask(seq_len):
    """Return a (1, 1, seq_len, seq_len) bool mask, True on and below diagonal."""
    # TODO: build a lower-triangular boolean causal mask of shape (1, 1, seq_len, seq_len)
    mask = torch.ones(seq_len, seq_len, dtype=torch.bool).tril(diagonal=0)

    return mask.unsqueeze(0).unsqueeze(0)

# Step 16 - combine_padding_and_causal_masks
import torch

def combine_padding_and_causal_masks(padding_mask, causal_mask):
    # TODO: combine a (B,1,1,L) padding mask with a (1,1,L,L) causal mask into (B,1,L,L).
    return padding_mask & causal_mask

# Step 17 - compute_raw_attention_scores
import torch

def compute_raw_attention_scores(query, key):
    """Compute raw attention scores Q @ K^T over the last two dimensions."""
    # TODO: matmul query with the transpose of key over the last two axes
    return query @ key.transpose(-1, -2)

# Step 18 - scale_attention_scores
import torch
import math

def scale_attention_scores(scores, d_k):
    # TODO: divide raw attention scores by sqrt(d_k) to stabilize softmax inputs
    return scores / math.sqrt(d_k)

# Step 19 - mask_attention_scores_with_neg_inf
import torch

def mask_attention_scores_with_neg_inf(scores, mask):
    """Set entries of scores where mask is False to -inf."""
    # TODO: replace blocked positions of scores with negative infinity
    return scores.masked_fill(~mask, float('-inf'))

# Step 20 - softmax_attention_weights
import torch

def softmax_attention_weights(masked_scores):
    # TODO: softmax over the last axis, zeroing rows that are entirely -inf
    weights = torch.softmax(masked_scores, dim=-1)

    return torch.nan_to_num(weights, nan=0.0)

# Step 21 - apply_attention_weights_to_values
import torch

def apply_attention_weights_to_values(attention_weights, value):
    """Multiply attention weights by the value matrix to produce context vectors."""
    # TODO: combine attention weights (..., Lq, Lk) with value (..., Lk, d_v)
    return attention_weights @ value

# Step 22 - scaled_dot_product_attention
import torch

def scaled_dot_product_attention(query, key, value, mask=None):
    """Run scaled dot-product attention; return (context, attention_weights)."""
    # TODO: chain raw scores, scale by sqrt(d_k), optionally mask, softmax, then mix values
    d_k = query.size(-1)

    scores = query @ key.transpose(-1, -2)
    scores = scores / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))

    weights = torch.softmax(scores, dim=-1)
    weights = torch.nan_to_num(weights, nan=0.0)

    context = weights @ value

    return context, weights

# Step 23 - split_last_dim_into_heads
import torch

def split_last_dim_into_heads(tensor, num_heads):
    # TODO: reshape (B, L, d_model) into (B, L, num_heads, d_model // num_heads)
    *batch_dims, d_model = tensor.shape
    head_dim = d_model // num_heads

    return tensor.view(*batch_dims, num_heads, head_dim)

# Step 24 - transpose_heads_before_sequence
import torch

def transpose_heads_before_sequence(split_tensor):
    # TODO: rearrange (B, L, num_heads, d_k) into (B, num_heads, L, d_k).
    return split_tensor.transpose(1, 2)

# Step 25 - merge_heads_back_to_model_dim
import torch

def merge_heads_back_to_model_dim(multi_head_tensor):
    # TODO: merge the head axis back into the feature axis to reconstruct d_model
    batch_size, num_heads, seq_len, head_dim = multi_head_tensor.shape

    return multi_head_tensor.transpose(1, 2).contiguous().view(batch_size, seq_len, num_heads * head_dim)

# Step 26 - apply_linear_projection
def apply_linear_projection(x, weight, bias):
    # TODO: return x @ weight^T + bias (bias may be None) with shape (..., out_features)
    out = x @ weight.transpose(-1, -2)

    if bias is not None:
        out = out + bias

    return out

# Step 27 - project_to_query_key_value
def project_to_query_key_value(x, w_q, b_q, w_k, b_k, w_v, b_v):
    # TODO: project x into separate query, key, and value tensors via three linear layers
    def _project(weight, bias):
        out = x @ weight.transpose(-1, -2)

        if bias is not None:
            out = out + bias
        
        return out

    q = _project(w_q, b_q)
    k = _project(w_k, b_k)
    v = _project(w_v, b_v)

    return q, k, v

# Step 28 - split_qkv_into_heads
import torch

def split_qkv_into_heads(q, k, v, num_heads):
    # TODO: split each of q, k, v into (B, num_heads, L, d_k) and return as a tuple
    def _transform(tensor: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = tensor.shape
        head_dim = d_model // num_heads

        return tensor.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

    return _transform(q), _transform(k), _transform(v)

# Step 29 - multi_head_scaled_dot_product_attention
import torch

def multi_head_scaled_dot_product_attention(q_h, k_h, v_h, mask=None):
    # TODO: run scaled dot-product attention over per-head Q, K, V and return (context, weights)
    d_k = q_h.size(-1)

    scores = q_h @ k_h.transpose(-1, -2)
    scores = scores / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))

    weights = torch.softmax(scores, dim=-1)
    weights = torch.nan_to_num(weights, nan=0.0)

    context = weights @ v_h

    return context, weights

# Step 30 - merge_heads_and_project_output
import torch

def merge_heads_and_project_output(context, w_o, b_o):
    # TODO: merge the head axis back into d_model and apply the output linear projection.
    batch_size, num_heads, seq_len, head_dim = context.shape
    merged = context.transpose(1, 2).contiguous().view(batch_size, seq_len, num_heads * head_dim)

    out = merged @ w_o.transpose(-1, -2)

    if b_o is not None:
        out = out + b_o

    return out

# Step 31 - assemble_multi_head_attention_forward
def assemble_multi_head_attention_forward(query, key, value, w_q, w_k, w_v, w_o, num_heads, mask=None):
    # TODO: project Q/K/V, split into heads, run scaled dot-product attention, merge heads, output projection.
    if isinstance(query, tuple):
        q_tensor, k_tensor, v_tensor = query
        w_q, w_k, w_v, w_o, num_heads, mask = key, value, w_q, w_k, w_v, w_o
        query, key, value = q_tensor, k_tensor, v_tensor

    if key is None:
        key = query
    if value is None:
        value = query
    
    q = query @ w_q.transpose(-1, -2)
    k = key @ w_k.transpose(-1, -2)
    v = value @ w_v.transpose(-1, -2)

    def _split(tensor: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = tensor.shape
        head_dim = d_model // num_heads

        return tensor.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

    q_h = _split(q)
    k_h = _split(k)
    v_h = _split(v)

    d_k = q_h.size(-1)

    scores = q_h @ k_h.transpose(-1, -2)
    scores = scores / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    weights = torch.softmax(scores, dim=-1)
    weights = torch.nan_to_num(weights, nan=0.0)

    context = weights @ v_h

    batch_size, _, seq_len, head_dim = context.shape
    merged = context.transpose(1, 2).contiguous().view(batch_size, seq_len, num_heads * head_dim)

    output = merged @ w_o.transpose(-1, -2)

    return output

# Step 32 - apply_ffn_first_linear_and_relu
def apply_ffn_first_linear_and_relu(x, w1, b1):
    # TODO: project x by w1, add b1, then apply a ReLU activation.
    hidden = x @ w1

    if b1 is not None:
        hidden = hidden + b1

    return torch.relu(hidden)

# Step 33 - apply_ffn_second_linear
import torch

def apply_ffn_second_linear(hidden, w2, b2):
    # TODO: project hidden (..., d_ff) back to (..., d_model) via w2 and b2.
    if w2.shape[0] != hidden.shape[-1]:
        w2 = w2.transpose(-1, -2)

    out = hidden @ w2

    if b2 is not None:
        out = out + b2

    return out

# Step 34 - position_wise_feed_forward_network
def position_wise_feed_forward_network(x, w1, b1, w2, b2):
    # TODO: compose the two FFN linears with a ReLU in between, returning shape (B, T, d_model).
    hidden = apply_ffn_first_linear_and_relu(x, w1, b1)

    return apply_ffn_second_linear(hidden, w2, b2)

# Step 35 - compute_layer_norm_mean_and_variance
import torch

def compute_layer_norm_mean_and_variance(x):
    # TODO: return (mean, variance) reduced over the last dim with shape (..., 1)
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)

    return mean, var

# Step 36 - normalize_and_scale_with_gamma_beta
import torch

def normalize_and_scale_with_gamma_beta(x, gamma, beta, eps=1e-5):
    # TODO: standardize x along the last axis then apply gamma and beta affine transform
    mean, var = compute_layer_norm_mean_and_variance(x)
    normalized = (x - mean) / torch.sqrt(var + eps)
    out = normalized * gamma + beta

    return out

# Step 37 - apply_residual_add_and_norm
import torch

def apply_residual_add_and_norm(residual_input, sublayer_output, gamma, beta, eps=1e-5):
    # TODO: combine the residual with the sublayer output and layer-normalize the result.
    x = residual_input + sublayer_output

    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)

    normalized = (x - mean) / torch.sqrt(var + eps)
    out = normalized * gamma + beta

    return out

# Step 38 - apply_dropout_with_keep_mask
def apply_dropout_with_keep_mask(x, keep_mask, keep_prob):
    # TODO: multiply x by the boolean keep_mask and rescale by 1/keep_prob.
    if keep_prob == 0.0:
        return torch.zeros_like(x)

    return x * keep_mask.to(x.dtype) / keep_prob

# Step 39 - encoder_layer_self_attention_sublayer
def encoder_layer_self_attention_sublayer(x, w_q, w_k, w_v, w_o, gamma, beta, num_heads, src_mask):
    # TODO: run multi-head self-attention on x and wrap with residual add-and-norm.
    result = assemble_multi_head_attention_forward(
        query=x,
        key=x,
        value=x,
        w_q=w_q,
        w_k=w_k,
        w_v=w_v,
        w_o=w_o,
        num_heads=num_heads,
        mask=src_mask
    )

    sublayer_output = result[0] if isinstance(result, tuple) else result

    out = apply_residual_add_and_norm(
        residual_input = x,
        sublayer_output=sublayer_output,
        gamma=gamma,
        beta=beta
    )

    return out

# Step 40 - encoder_layer_feed_forward_sublayer
def encoder_layer_feed_forward_sublayer(x, w1, b1, w2, b2, gamma, beta):
    # TODO: run the position-wise FFN on x and wrap it with residual add-and-norm.
    h = x @ w1

    if b1 is not None:
        h = h + b1

    h = torch.relu(h)
    sublayer_output = h @ w2

    if b2 is not None:
        sublayer_output = sublayer_output + b2

    out = apply_residual_add_and_norm(
        residual_input=x,
        sublayer_output=sublayer_output,
        gamma=gamma,
        beta=beta
    )

    return out

# Step 41 - assemble_encoder_layer
def assemble_encoder_layer(x, layer_params, num_heads, src_mask):
    # TODO: chain the self-attention sublayer and the feed-forward sublayer using layer_params.
    if isinstance(layer_params, dict):
        w_q = layer_params.get('w_q')
        w_k = layer_params.get('w_k')
        w_v = layer_params.get('w_v')
        w_o = layer_params.get('w_o')

        gamma1 = layer_params.get('gamma1', layer_params.get('ln1_gamma'))
        beta1 = layer_params.get('beta1', layer_params.get('ln1_beta'))

        w1 = layer_params.get('w1')
        b1 = layer_params.get('b1')
        w2 = layer_params.get('w2')
        b2 = layer_params.get('b2')

        gamma2 = layer_params.get('gamma2', layer_params.get('ln2_gamma'))
        beta2 = layer_params.get('beta2', layer_params.get('ln2_beta'))

    elif isinstance(layer_params, (tuple, list)):
        w_q, w_k, w_v, w_o, gamma1, beta1, w1, b1, w2, b2, gamma2, beta2 = layer_params

    else:
        raise TypeError("layer_params must be a dictionary, tuple, or list.")

    batch_size, seq_len, d_model = x.shape
    head_dim = d_model // num_heads

    q = x @ w_q.transpose(-1, -2)
    k = x @ w_k.transpose(-1, -2)
    v = x @ w_v.transpose(-1, -2)

    def _split(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

    q_h, k_h, v_h = _split(q), _split(k), _split(v)
    scores = (q_h @ k_h.transpose(-1, -2)) / math.sqrt(head_dim)

    if src_mask is not None:
        scores = scores.masked_fill(~src_mask, float('-inf'))

    weights = torch.softmax(scores, dim=-1)
    weights = torch.nan_to_num(weights, nan=0.0)
    context = weights @ v_h

    merged = context.transpose(1, 2).contiguous().view(batch_size, seq_len, num_heads * head_dim)
    attn_out = merged @ w_o.transpose(-1, -2)

    x_res1 = x + attn_out
    mean1 = x_res1.mean(dim=-1, keepdim=True)
    var1 = x_res1.var(dim=-1, unbiased=False, keepdim=True)
    norm1 = (x_res1 - mean1) / torch.sqrt(var1 + 1e-5)
    
    if gamma1 is not None:
        norm1 = norm1 * gamma1

    if beta1 is not None:
        norm1 = norm1 + beta1

    x = norm1
    h = x @ w1

    if b1 is not None:
        h = h + b1

    h = torch.relu(h)
    
    ffn_out = h @ w2

    if b2 is not None:
        ffn_out = ffn_out + b2

    x_res2 = x + ffn_out
    mean2 = x_res2.mean(dim=-1, keepdim=True)
    var2 = x_res2.var(dim=-1, unbiased=False, keepdim=True)
    norm2 = (x_res2 - mean2) / torch.sqrt(var2 + 1e-5)
    
    if gamma2 is not None:
        norm2 = norm2 * gamma2

    if beta2 is not None:
        norm2 = norm2 + beta2
        
    x = norm2

    return x

# Step 42 - stack_encoder_layers
def stack_encoder_layers(x, encoder_layer_params_list, num_heads, src_mask):
    # TODO: sequentially apply each encoder layer to the running hidden state and return the final tensor.
    for layer_params in encoder_layer_params_list:
        x = assemble_encoder_layer(
            x=x,
            layer_params=layer_params,
            num_heads=num_heads,
            src_mask=src_mask
        )

    return x

# Step 43 - decoder_layer_masked_self_attention_sublayer
import torch

def decoder_layer_masked_self_attention_sublayer(y, w_q, w_k, w_v, w_o, gamma, beta, num_heads, tgt_mask):
    # TODO: run masked multi-head self-attention on y and wrap with residual add-and-norm.
    batch_size, seq_len, d_model = y.shape
    head_dim = d_model // num_heads

    q = y @ w_q.transpose(-1, -2)
    k = y @ w_k.transpose(-1, -2)
    v = y @ w_v.transpose(-1, -2)

    def _split(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

    q_h, k_h, v_h = _split(q), _split(k), _split(v)
    scores = (q_h @ k_h.transpose(-1, -2)) / math.sqrt(head_dim)

    if tgt_mask is not None:
        scores = scores.masked_fill(~tgt_mask, float('-inf'))

    weights = torch.softmax(scores, dim=-1)
    weights = torch.nan_to_num(weights, nan=0.0)
    context = weights @ v_h

    merged = context.transpose(1, 2).contiguous().view(batch_size, seq_len, num_heads * head_dim)
    attn_out = merged @ w_o.transpose(-1, -2)

    y_res = y + attn_out
    mean = y_res.mean(dim=-1, keepdim=True)
    var = y_res.var(dim=-1, unbiased=False, keepdim=True)
    norm = (y_res - mean) / torch.sqrt(var + 1e-5)

    if gamma is not None:
        norm = norm * gamma

    if beta is not None:
        norm = norm + beta

    return norm

# Step 44 - decoder_layer_cross_attention_sublayer
import torch

def decoder_layer_cross_attention_sublayer(y, encoder_output, w_q, w_k, w_v, w_o, gamma, beta, num_heads, src_mask):
    # TODO: run multi-head cross-attention (Q from y, K/V from encoder_output) and wrap with add-and-norm
    batch_size, tgt_seq_len, d_model = y.shape
    _, src_seq_len, _ = encoder_output.shape
    head_dim = d_model // num_heads

    q = y @ w_q.transpose(-1, -2)
    k = encoder_output @ w_k.transpose(-1, -2)
    v = encoder_output @ w_v.transpose(-1, -2)

    def _split_q(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.view(batch_size, tgt_seq_len, num_heads, head_dim).transpose(1, 2)
    
    def _split_kv(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.view(batch_size, src_seq_len, num_heads, head_dim).transpose(1, 2)

    q_h = _split_q(q)
    k_h = _split_kv(k)
    v_h = _split_kv(v)

    scores = (q_h @ k_h.transpose(-1, -2)) / math.sqrt(head_dim)

    if src_mask is not None:
        if src_mask.ndim == 2:
            mask = src_mask.unsqueeze(1).unsqueeze(2)
        else:
            mask = src_mask

        scores = scores.masked_fill(~mask, float('-inf'))

    weights = torch.softmax(scores, dim=-1)
    weights = torch.nan_to_num(weights, nan=0.0)
    context = weights @ v_h

    merged = context.transpose(1, 2).contiguous().view(batch_size, tgt_seq_len, num_heads * head_dim)
    attn_out = merged @ w_o.transpose(-1, -2)

    y_res = y + attn_out
    mean = y_res.mean(dim=-1, keepdim=True)
    var = y_res.var(dim=-1, unbiased=False, keepdim=True)
    norm = (y_res - mean) / torch.sqrt(var + 1e-5)

    if gamma is not None:
        norm = norm * gamma

    if beta is not None:
        norm = norm + beta

    return norm

# Step 45 - decoder_layer_feed_forward_sublayer
import torch

def decoder_layer_feed_forward_sublayer(y, w1, b1, w2, b2, gamma, beta):
    # TODO: run the position-wise FFN on y and wrap it with residual add-and-norm
    h = y @ w1

    if b1 is not None:
        h = h + b1

    h = torch.relu(h)

    ffn_out = h @ w2

    if b2 is not None:
        ffn_out = ffn_out + b2

    y_res = y + ffn_out
    mean = y_res.mean(dim=-1, keepdim=True)
    var = y_res.var(dim=-1, unbiased=False, keepdim=True)
    norm = (y_res - mean) / torch.sqrt(var + 1e-5)

    if gamma is not None:
        norm = norm * gamma
    if beta is not None:
        norm = norm + beta

    return norm

# Step 46 - assemble_decoder_layer
def assemble_decoder_layer(y, encoder_output, layer_params, num_heads, src_mask, tgt_mask):
    """Run a full decoder layer: masked self-attention, cross-attention, then FFN.

    layer_params keys (all torch tensors):
      masked self-attention : w_q_self, w_k_self, w_v_self, w_o_self, self_gamma, self_beta
      cross-attention       : w_q_cross, w_k_cross, w_v_cross, w_o_cross, cross_gamma, cross_beta
      feed-forward          : w1, b1, w2, b2, ffn_gamma, ffn_beta
    """
    # TODO: chain the three decoder sublayers using params from layer_params.
    y = decoder_layer_masked_self_attention_sublayer(
      y=y,
      w_q=layer_params['w_q_self'],
      w_k=layer_params['w_k_self'],
      w_v=layer_params['w_v_self'],
      w_o=layer_params['w_o_self'],
      gamma=layer_params['self_gamma'],
      beta=layer_params['self_beta'],
      num_heads=num_heads,
      tgt_mask=tgt_mask
    )

    y = decoder_layer_cross_attention_sublayer(
        y=y,
        encoder_output=encoder_output,
        w_q=layer_params['w_q_cross'],
        w_k=layer_params['w_k_cross'],
        w_v=layer_params['w_v_cross'],
        w_o=layer_params['w_o_cross'],
        gamma=layer_params['cross_gamma'],
        beta=layer_params['cross_beta'],
        num_heads=num_heads,
        src_mask=src_mask
    )

    y = decoder_layer_feed_forward_sublayer(
        y=y,
        w1=layer_params['w1'],
        b1=layer_params['b1'],
        w2=layer_params['w2'],
        b2=layer_params['b2'],
        gamma=layer_params['ffn_gamma'],
        beta=layer_params['ffn_beta']
    )

    return y

# Step 47 - stack_decoder_layers
def stack_decoder_layers(y, encoder_output, decoder_layer_params_list, num_heads, src_mask, tgt_mask):
    # TODO: sequentially apply each decoder layer to the running target hidden state.
    for layer_params in decoder_layer_params_list:
        y = assemble_decoder_layer(
            y=y,
            encoder_output=encoder_output,
            layer_params=layer_params,
            num_heads=num_heads,
            src_mask=src_mask,
            tgt_mask=tgt_mask
        )

    return y

# Step 48 - apply_final_output_projection
def apply_final_output_projection(decoder_output, output_projection_weight, output_projection_bias=None):
    # TODO: project decoder hidden states (B, T, D) to vocabulary logits (B, T, V).
    d_model = decoder_output.shape[-1]

    if output_projection_weight.shape[-1] == d_model:
        logits = decoder_output @ output_projection_weight.transpose(-1, -2)
    else:
        logits = decoder_output @ output_projection_weight
    
    if output_projection_bias is not None:
        logits = logits + output_projection_bias

    return logits

# Step 49 - tie_output_projection_to_token_embeddings
import torch

def tie_output_projection_to_token_embeddings(token_embedding_weight):
    """Return an output projection weight that shares storage with token_embedding_weight.

    Input shape: (vocab_size, d_model). Output shape: (d_model, vocab_size).
    """
    # TODO: return an output projection weight tied to the token embedding matrix
    return token_embedding_weight.t()

# Step 50 - apply_log_softmax_over_vocab
import torch
import torch.nn.functional as F

def apply_log_softmax_over_vocab(logits):
    # TODO: Convert decoder logits (B, T, V) into log probabilities over the vocabulary axis.
    return F.log_softmax(logits, dim=-1)

# Step 51 - run_transformer_forward
def run_transformer_forward(src_ids, tgt_ids, model_params, num_heads, pad_id):
    # TODO: embed src+tgt, add PE, build masks, run encoder/decoder, project to log probs.
    token_embedding = model_params.get(
        "token_embedding",
        model_params.get("src_embed", model_params.get("embedding")),
    )
    tgt_embedding = model_params.get("tgt_embed", token_embedding)

    if isinstance(token_embedding, torch.nn.Embedding):
        d_model = token_embedding.embedding_dim
    else:
        d_model = token_embedding.shape[-1]

    def _embed(ids, table):
        if ids.is_floating_point():
            return ids
        if isinstance(table, torch.nn.Embedding):
            return table(ids)
        if isinstance(table, torch.Tensor):
            return F.embedding(ids, table, padding_idx=pad_id)
        if callable(table):
            return table(ids)
        return ids

    def _add_sinusoidal_pe(t):
        seq_len, dm = t.size(1), t.size(-1)
        pos = torch.arange(seq_len, dtype=t.dtype, device=t.device).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, dm, 2, dtype=t.dtype, device=t.device)
            * (-math.log(10000.0) / dm)
        )
        pe = torch.zeros(seq_len, dm, dtype=t.dtype, device=t.device)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)[:, : dm // 2]
        return t + pe.unsqueeze(0)

    x = _embed(src_ids, token_embedding) * math.sqrt(d_model)
    y = _embed(tgt_ids, tgt_embedding) * math.sqrt(d_model)

    pos_enc = model_params.get("pos_encoder")
    pos_dec = model_params.get("pos_decoder", pos_enc)
    x = pos_enc(x) if pos_enc is not None else _add_sinusoidal_pe(x)
    y = pos_dec(y) if pos_dec is not None else _add_sinusoidal_pe(y)

    if not src_ids.is_floating_point():
        src_mask = src_ids != pad_id
    else:
        src_mask = torch.ones(src_ids.shape[:2], dtype=torch.bool, device=src_ids.device)

    batch_size, tgt_seq_len = tgt_ids.shape[:2]
    device = tgt_ids.device
    causal_mask = torch.tril(
        torch.ones((tgt_seq_len, tgt_seq_len), dtype=torch.bool, device=device)
    )
    if not tgt_ids.is_floating_point():
        tgt_pad_mask = (tgt_ids != pad_id).unsqueeze(1)
        tgt_mask = causal_mask & tgt_pad_mask
    else:
        tgt_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)

    encoder_layers = model_params.get(
        "encoder_layers", model_params.get("encoder_layer_params_list")
    )
    decoder_layers = model_params.get(
        "decoder_layers", model_params.get("decoder_layer_params_list")
    )

    encoder_output = stack_encoder_layers(x, encoder_layers, num_heads, src_mask)
    decoder_output = stack_decoder_layers(
        y, encoder_output, decoder_layers, num_heads, src_mask, tgt_mask
    )

    output_proj = model_params.get(
        "output_projection", model_params.get("output_projection_weight")
    )
    
    if output_proj is None:
        if isinstance(tgt_embedding, torch.nn.Embedding):
            output_proj = tgt_embedding.weight.t()
        elif isinstance(tgt_embedding, torch.Tensor):
            output_proj = tgt_embedding.t()

    logits = apply_final_output_projection(
        decoder_output=decoder_output,
        output_projection_weight=output_proj,
        output_projection_bias=model_params.get("output_projection_bias"),
    )

    return apply_log_softmax_over_vocab(logits)

# Step 52 - init_encoder_layer_parameters
import torch
import math

def init_encoder_layer_parameters(d_model, num_heads, d_ff):
    """Return a dict of leaf tensors with requires_grad=True for one encoder layer."""
    # TODO: allocate w_q, w_k, w_v, w_o, w1, b1, w2, b2, attn_gamma, attn_beta, ffn_gamma, ffn_beta.
    if d_model % num_heads != 0:
        raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

    dtype = torch.float32

    def _xavier(fan_in, fan_out, shape):
        bound = math.sqrt(6.0 / (fan_in + fan_out))
        t = torch.empty(*shape, dtype=dtype).uniform_(-bound, bound)
        return t.requires_grad_(True)

    def _zeros(*shape):
        return torch.zeros(*shape, dtype=dtype, requires_grad=True)

    def _ones(*shape):
        return torch.ones(*shape, dtype=dtype, requires_grad=True)

    return {
        "w_q": _xavier(d_model, d_model, (d_model, d_model)),
        "w_k": _xavier(d_model, d_model, (d_model, d_model)),
        "w_v": _xavier(d_model, d_model, (d_model, d_model)),
        "w_o": _xavier(d_model, d_model, (d_model, d_model)),

        "w1": _xavier(d_model, d_ff, (d_model, d_ff)),
        "b1": _zeros(d_ff),
        "w2": _xavier(d_ff, d_model, (d_ff, d_model)),
        "b2": _zeros(d_model),

        "attn_gamma": _ones(d_model),
        "attn_beta": _zeros(d_model),
        "ffn_gamma": _ones(d_model),
        "ffn_beta": _zeros(d_model),
    }

# Step 53 - init_decoder_layer_parameters
import torch

def init_decoder_layer_parameters(d_model, num_heads, d_ff):
    # TODO: return a dict of requires_grad tensors for one decoder layer
    if d_model % num_heads != 0:
        raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

    dtype = torch.float32

    def _xavier(fan_in, fan_out, shape):
        bound = math.sqrt(6.0 / (fan_in + fan_out))
        t = torch.empty(*shape, dtype=dtype).uniform_(-bound, bound)
        return t.requires_grad_(True)

    def _zeros(*shape):
        return torch.zeros(*shape, dtype=dtype, requires_grad=True)

    def _ones(*shape):
        return torch.ones(*shape, dtype=dtype, requires_grad=True)

    def _square():
        return _xavier(d_model, d_model, (d_model, d_model))

    return {
        "w_q_self": _square(),
        "w_k_self": _square(),
        "w_v_self": _square(),
        "w_o_self": _square(),

        "w_q_cross": _square(),
        "w_k_cross": _square(),
        "w_v_cross": _square(),
        "w_o_cross": _square(),

        "w1": _xavier(d_model, d_ff, (d_model, d_ff)),
        "b1": _zeros(d_ff),
        "w2": _xavier(d_ff, d_model, (d_ff, d_model)),
        "b2": _zeros(d_model),

        "self_gamma": _ones(d_model),
        "self_beta": _zeros(d_model),
        "cross_gamma": _ones(d_model),
        "cross_beta": _zeros(d_model),
        "ffn_gamma": _ones(d_model),
        "ffn_beta": _zeros(d_model),
    }

# Step 54 - init_embedding_and_projection_parameters
import torch

def init_embedding_and_projection_parameters(vocab_size, d_model, tie_weights=True):
    """Allocate src/tgt embeddings and output projection (optionally tied)."""
    # TODO: allocate three (vocab_size, d_model) tensors with requires_grad=True
    dtype = torch.float32
    std = d_model ** -0.5

    def _embed_table():
        t = torch.empty(vocab_size, d_model, dtype=dtype).normal_(0.0, std)
        return t.requires_grad_(True)

    src_embedding = _embed_table()
    tgt_embedding = _embed_table()

    output_projection = tgt_embedding if tie_weights else _embed_table()

    return {
        "src_embedding": src_embedding,
        "tgt_embedding": tgt_embedding,
        "output_projection": output_projection,
    }

# Step 55 - collect_model_parameters_into_list
import torch

def collect_model_parameters_into_list(encoder_layer_params, decoder_layer_params, embedding_params):
    # TODO: walk the encoder, decoder, and embedding dicts and return a flat deduped list of tensors
    params = []
    seen = set()

    def _walk(obj):
        if isinstance(obj, torch.Tensor):
            if id(obj) not in seen:
                seen.add(id(obj))
                params.append(obj)
        elif isinstance(obj, dict):
            for value in obj.values():
                _walk(value)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                _walk(item)

    _walk(encoder_layer_params)
    _walk(decoder_layer_params)
    _walk(embedding_params)
    
    return params

# Step 56 - shift_targets_right_with_start_token
def shift_targets_right_with_start_token(target_ids, start_token_id):
    # TODO: prepend start_token_id and drop the last column so output shape matches target_ids
    start = torch.full(
        (*target_ids.shape[:-1], 1),
        start_token_id,
        dtype=target_ids.dtype,
        device=target_ids.device,
    )

    return torch.cat([start, target_ids[..., :-1]], dim=-1)

# Step 57 - compute_noam_learning_rate
def compute_noam_learning_rate(step, d_model, warmup_steps):
    # TODO: return the Noam warmup learning rate for the given step.
    step = max(step, 1)

    return (d_model ** -0.5) * min(step ** -0.5, step * warmup_steps ** -1.5)

# Step 58 - build_uniform_smoothing_distribution
import torch

def build_uniform_smoothing_distribution(shape, vocab_size, epsilon):
    # TODO: return a float tensor of `shape` filled with epsilon / (vocab_size - 2).
    if vocab_size <= 2:
        raise ValueError(f"vocab_size must be > 2 for label smoothing, got {vocab_size}")

    fill_value = epsilon / (vocab_size - 2)

    return torch.full(tuple(shape) if not isinstance(shape, int) else (shape,),
        fill_value, dtype=torch.float32)

# Step 59 - set_confidence_on_gold_tokens
import torch

def set_confidence_on_gold_tokens(smoothed_distribution, gold_token_ids, confidence):
    """Place confidence mass at gold-token positions of a smoothed target distribution."""
    # TODO: write the confidence value at each gold token id along the vocab axis
    result = smoothed_distribution.clone()
    index = gold_token_ids.long().unsqueeze(-1)
    result.scatter_(-1, index, confidence)

    return result

# Step 60 - zero_pad_column_and_pad_token_rows
import torch

def zero_pad_column_and_pad_token_rows(smoothed_distribution, gold_token_ids, pad_id):
    # TODO: zero the pad column and the rows where the gold token equals pad_id
    result = smoothed_distribution.clone()
    result[..., pad_id] = 0.0
    pad_rows = (gold_token_ids == pad_id).unsqueeze(-1)

    return result.masked_fill(pad_rows, 0.0)

# Step 61 - compute_label_smoothed_kl_loss
import torch

def compute_label_smoothed_kl_loss(log_probabilities, smoothed_distribution):
    """Return the summed KL loss over all (batch, time, vocab) entries."""
    # TODO: combine log_probabilities with the smoothed target distribution into a scalar loss
    return (smoothed_distribution * -log_probabilities).sum()

# Step 62 - average_loss_over_non_pad_tokens
import torch

def average_loss_over_non_pad_tokens(total_loss, gold_token_ids, pad_id):
    # TODO: divide total_loss by the count of non-pad tokens in gold_token_ids
    num_tokens = (gold_token_ids != pad_id).sum().clamp(min=1)

    return total_loss / num_tokens.to(total_loss.dtype)

# Step 63 - compute_token_accuracy_ignoring_pad
import torch

def compute_token_accuracy_ignoring_pad(log_probabilities, gold_token_ids, pad_id):
    # TODO: argmax over vocab, compare to gold, average over non-pad positions only
    predictions = log_probabilities.argmax(dim=-1)
    non_pad = gold_token_ids != pad_id
    correct = (predictions == gold_token_ids) & non_pad
    num_tokens = non_pad.sum().clamp(min=1)

    return correct.sum().float() / num_tokens.float()

# Step 64 - initialize_adam_optimizer_state
import torch

def initialize_adam_optimizer_state(parameter_list):
    """Allocate Adam m, v zero buffers and a step counter t=0."""
    # TODO: allocate zero buffers for first and second moments, plus step counter
    return {
        "m": [torch.zeros_like(p, requires_grad=False) for p in parameter_list],
        "v": [torch.zeros_like(p, requires_grad=False) for p in parameter_list],
        "t": 0,
    }

# Step 65 - update_adam_first_moment
import torch

def update_adam_first_moment(m_prev, grad, beta1):
    """Return m_t = beta1 * m_prev + (1 - beta1) * grad."""
    # TODO: apply the Adam first-moment EMA update and return the new tensor
    return beta1 * m_prev + (1 - beta1) * grad

# Step 66 - update_adam_second_moment
import torch

def update_adam_second_moment(v_prev, grad, beta2):
    """Return v_t = beta2 * v_prev + (1 - beta2) * grad ** 2."""
    # TODO: apply Adam's EMA update for the second moment of the gradient
    return beta2 * v_prev + (1 - beta2) * grad * grad

# Step 67 - apply_adam_bias_correction
import torch

def apply_adam_bias_correction(m_t, v_t, beta1, beta2, step):
    """Return bias-corrected (m_hat, v_hat) for Adam at the given step."""
    # TODO: divide each moment by (1 - beta**step) using its respective beta
    m_hat = m_t / (1 - beta1 ** step)
    v_hat = v_t / (1 - beta2 ** step)

    return m_hat, v_hat

# Step 68 - compute_adam_parameter_update
import torch

def compute_adam_parameter_update(m_hat, v_hat, learning_rate, epsilon):
    """Return delta = learning_rate * m_hat / (sqrt(v_hat) + epsilon); the caller subtracts it."""
    # TODO: compute the Adam step from the bias-corrected moments without tracking gradients
    with torch.no_grad():
        return learning_rate * m_hat / (v_hat.sqrt() + epsilon)

# Step 69 - apply_adam_step_to_all_parameters
import torch

def apply_adam_step_to_all_parameters(parameter_list, optimizer_state, learning_rate, beta1=0.9, beta2=0.98, epsilon=1e-9):
    # TODO: increment t, then for each param with a grad update m, v, bias-correct, and subtract delta in place.
    optimizer_state["t"] += 1
    step = optimizer_state["t"]

    with torch.no_grad():
        for i, param in enumerate(parameter_list):
            if param.grad is None:
                continue
            grad = param.grad

            m_t = update_adam_first_moment(optimizer_state["m"][i], grad, beta1)
            v_t = update_adam_second_moment(optimizer_state["v"][i], grad, beta2)
            optimizer_state["m"][i] = m_t
            optimizer_state["v"][i] = v_t

            m_hat, v_hat = apply_adam_bias_correction(m_t, v_t, beta1, beta2, step)
            delta = compute_adam_parameter_update(m_hat, v_hat, learning_rate, epsilon)
            param.sub_(delta)

    return optimizer_state

# Step 70 - zero_all_parameter_gradients
import torch

def zero_all_parameter_gradients(parameter_list):
    """Clear the .grad of every parameter tensor before the next backward pass."""
    # TODO: clear the accumulated gradient on every parameter tensor in the list
    for param in parameter_list:
        param.grad = None

# Step 71 - compute_batch_training_loss
def compute_batch_training_loss(src_batch, tgt_batch, model_params, config):
    """One teacher-forced forward pass -> label-smoothed KL loss averaged over non-pad tokens.

    model_params (read only; the same dict run_transformer_forward takes):
      token_embedding, output_projection, encoder_layers, decoder_layers
    config keys:
      pad_id, start_id, vocab_size, smoothing, num_heads
    """
    # TODO: shift targets right, run the forward pass, build smoothed targets, and average the KL loss over non-pad tokens.
    pad_id = config["pad_id"]
    start_id = config["start_id"]
    vocab_size = config["vocab_size"]
    smoothing = config["smoothing"]
    num_heads = config["num_heads"]

    decoder_input = shift_targets_right_with_start_token(tgt_batch, start_id)

    src_table = model_params.get("token_embedding", model_params.get("src_embedding"))
    tgt_table = model_params.get("tgt_embedding", src_table)
    d_model = src_table.shape[-1]

    def _add_sinusoidal_pe(t):
        seq_len, dm = t.size(1), t.size(-1)
        pos = torch.arange(seq_len, dtype=t.dtype, device=t.device).unsqueeze(1)
        div = torch.exp(torch.arange(0, dm, 2, dtype=t.dtype, device=t.device)
                        * (-math.log(10000.0) / dm))
        pe = torch.zeros(seq_len, dm, dtype=t.dtype, device=t.device)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)[:, : dm // 2]
        return t + pe.unsqueeze(0)

    x = _add_sinusoidal_pe(F.embedding(src_batch, src_table, padding_idx=pad_id) * math.sqrt(d_model))
    y = _add_sinusoidal_pe(F.embedding(decoder_input, tgt_table, padding_idx=pad_id) * math.sqrt(d_model))

    src_mask = (src_batch != pad_id)[:, None, None, :]
    tgt_len = decoder_input.size(1)
    causal = torch.tril(torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=decoder_input.device))
    tgt_mask = causal[None, None, :, :] & (decoder_input != pad_id)[:, None, None, :]

    encoder_layers = model_params.get("encoder_layers", model_params.get("encoder_layer_params_list"))
    decoder_layers = model_params.get("decoder_layers", model_params.get("decoder_layer_params_list"))
    encoder_output = stack_encoder_layers(x, encoder_layers, num_heads, src_mask)
    decoder_output = stack_decoder_layers(y, encoder_output, decoder_layers, num_heads, src_mask, tgt_mask)

    output_proj = model_params.get("output_projection", model_params.get("output_projection_weight"))

    if output_proj is None:
        output_proj = tgt_table.t()

    logits = apply_final_output_projection(
        decoder_output=decoder_output,
        output_projection_weight=output_proj,
        output_projection_bias=model_params.get("output_projection_bias"),
    )

    log_probs = apply_log_softmax_over_vocab(logits)

    smoothed = build_uniform_smoothing_distribution(
        log_probs.shape, vocab_size, smoothing
    ).to(device=log_probs.device, dtype=log_probs.dtype)

    smoothed = set_confidence_on_gold_tokens(smoothed, tgt_batch, 1.0 - smoothing)
    smoothed = zero_pad_column_and_pad_token_rows(smoothed, tgt_batch, pad_id)

    total_loss = compute_label_smoothed_kl_loss(log_probs, smoothed)
    
    return average_loss_over_non_pad_tokens(total_loss, tgt_batch, pad_id)

# Step 72 - run_training_step_with_backprop
import torch

def run_training_step_with_backprop(src_batch, tgt_batch, parameter_list, model_params, optimizer_state, step_number, config):
    """Run one training iteration: zero grads, forward, backward, Noam LR, Adam step.

    Returns the scalar loss value for the step as a Python float.
    """
    # TODO: zero grads, compute loss, backward, look up Noam LR, apply Adam step
    encoder_aliases = {
        "gamma1": "attn_gamma", "beta1": "attn_beta",
        "gamma2": "ffn_gamma",  "beta2": "ffn_beta",
    }

    def _alias(layers, aliases):
        if not isinstance(layers, (list, tuple)):
            return layers
        out = []
        
        for layer in layers:
            if isinstance(layer, dict):
                layer = dict(layer)
                for new, old in aliases.items():
                    if new not in layer and old in layer:
                        layer[new] = layer[old]
            out.append(layer)
        return out

    mp = dict(model_params)

    for key in ("encoder_layers", "encoder_layer_params_list"):
        if key in mp:
            mp[key] = _alias(mp[key], encoder_aliases)

    zero_all_parameter_gradients(parameter_list)

    loss = compute_batch_training_loss(src_batch, tgt_batch, mp, config)
    loss.backward()

    d_model = config.get("d_model")

    if d_model is None:
        embedding = model_params.get("token_embedding", model_params.get("src_embedding"))
        d_model = embedding.shape[-1]

    warmup_steps = config.get("warmup_steps", 4000)
    learning_rate = compute_noam_learning_rate(step_number, d_model, warmup_steps)

    apply_adam_step_to_all_parameters(
        parameter_list,
        optimizer_state,
        learning_rate,
        beta1=config.get("beta1", 0.9),
        beta2=config.get("beta2", 0.98),
        epsilon=config.get("adam_epsilon", 1e-9),
    )

    return loss.item()

# Step 73 - run_training_loop_for_steps
def run_training_loop_for_steps(batches, parameter_list, model_params, optimizer_state, num_steps, config):
    """Run num_steps training iterations, cycling through batches, and return per-step losses."""
    # TODO: iterate for num_steps steps, calling run_training_step_with_backprop each time
    batches = list(batches)

    if num_steps > 0 and not batches:
        raise ValueError("batches must contain at least one (src, tgt) pair")

    losses = []

    for i in range(num_steps):
        batch = batches[i % len(batches)]

        if isinstance(batch, dict):
            src, tgt = batch["src"], batch["tgt"]
        else:
            src, tgt = batch

        loss = run_training_step_with_backprop(
            src, tgt, parameter_list, model_params, optimizer_state, i + 1, config
        )

        losses.append(loss)

    return losses

# Step 74 - pick_next_token_by_argmax
import torch

def pick_next_token_by_argmax(final_step_logits):
    """Greedy: return argmax token id per batch row.

    final_step_logits: FloatTensor of shape (batch, vocab_size)
    returns: LongTensor of shape (batch,)
    """
    # TODO: pick the next greedy token id by taking the argmax over the vocab axis
    return final_step_logits.argmax(dim=-1)

# Step 75 - compute_length_penalty
def compute_length_penalty(sequence_length, alpha):
    # TODO: return the Google NMT length penalty for the given sequence_length and alpha.
    return ((5 + sequence_length) / 6) ** alpha

# Step 76 - compute_candidate_scores
import torch

def compute_candidate_scores(beam_scores, next_token_log_probs):
    # TODO: add each beam's running log-prob to its row of next-token log probs.
    return beam_scores.unsqueeze(-1) + next_token_log_probs

# Step 77 - select_top_k_candidates
import torch

def select_top_k_candidates(candidate_scores, k):
    # TODO: pick the top k (beam_index, token_id, score) triples from candidate_scores
    vocab_size = candidate_scores.shape[-1]
    k = min(k, candidate_scores.numel())

    top_scores, flat_indices = candidate_scores.reshape(-1).topk(k)
    beam_indices = torch.div(flat_indices, vocab_size, rounding_mode="floor")
    token_ids = flat_indices % vocab_size

    return {
        "beam_indices": beam_indices,
        "token_ids": token_ids,
        "scores": top_scores,
    }

# Step 78 - append_tokens_to_beam_sequences
import torch

def append_tokens_to_beam_sequences(beam_sequences, beam_indices, token_ids):
    # TODO: gather parent beam rows and append the new token ids as the last column
    device = beam_sequences.device
    beam_indices = torch.as_tensor(beam_indices, device=device).long().reshape(-1)
    token_ids = torch.as_tensor(token_ids, device=device).to(beam_sequences.dtype).reshape(-1, 1)

    parents = beam_sequences.index_select(0, beam_indices)
    
    return torch.cat([parents, token_ids], dim=-1)

# Step 79 - mark_finished_beams
import torch

def mark_finished_beams(token_ids, finished_flags, end_token_id):
    # TODO: return updated boolean finished flags for each beam given the new token ids
    token_ids = torch.as_tensor(token_ids).reshape(-1)
    finished_flags = torch.as_tensor(finished_flags, device=token_ids.device).bool().reshape(-1)
    
    return finished_flags | (token_ids == end_token_id)

# Step 80 - select_best_finished_beam (not yet solved)
# TODO: implement

