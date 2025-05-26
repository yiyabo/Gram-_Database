#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Global-Feature Dual-Path VAE  for AMP sequence generation
--------------------------------------------------------
• Phase-A  : train auxiliary Transformer-AE   → aux_encoder.h5
• Phase-B  : build dual-path β-VAE + global feature enhancer
             (gated, KL-annealed, pad-masked reconstruction)
"""

# ────────────────────────────────────────────────────────────
# Imports & Env
# ────────────────────────────────────────────────────────────
import os, glob, pickle, subprocess, json, re, math, argparse, sys
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (Input, Dense, Embedding, Bidirectional, GRU,
                                     TimeDistributed, MultiHeadAttention,
                                     LayerNormalization, Dropout,
                                     GlobalAveragePooling1D, Concatenate,
                                     Add, Lambda, RepeatVector)
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import (ReduceLROnPlateau, EarlyStopping,
                                        ModelCheckpoint, Callback)
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
# ── auto GPU select / MPS check ─────────────────────────────────────────
if sys.platform == 'linux': # <-- Check for Linux
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        try:
            q = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
                text=True).strip().split("\n")
            gpu = min(range(len(q)), key=lambda i:int(q[i] or 0))
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
            print(f"[AutoGPU] selected GPU:{gpu}  mem_used:{q[gpu]} MiB")
        except Exception as e:
            print("[AutoGPU] fallback:", e)
elif sys.platform == 'darwin': # <-- Check for macOS
     # On macOS, TensorFlow usually uses MPS automatically if available
     # No need to explicitly set CUDA_VISIBLE_DEVICES
     print("[Platform] macOS detected. TensorFlow will attempt to use MPS if available.")
     # Optionally check if MPS is actually available (optional)
     # try:
     #     if tf.config.list_physical_devices('GPU'): # In TF >= 2.5, MPS is treated as GPU
     #         print("[MPS] Metal Performance Shaders (MPS) device found.")
     #     else:
     #         print("[MPS] No MPS device found. Will use CPU.")
     # except Exception as e:
     #     print("[MPS Check Error]", e)

# Setting memory growth might be beneficial (or harmless) for both CUDA and MPS
try:
    for g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)
    if tf.config.list_physical_devices("GPU"):
         print("[GPU/MPS] Memory growth enabled for detected GPU/MPS devices.")
except Exception as e:
    print("[Memory Growth Setup Error]", e)

np.random.seed(42); tf.random.set_seed(42)

# ────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────
class Config:
    latent_dim        = 128
    hidden_dim        = 128
    embedding_dim     = 128
    split_len         = 3 # k-mer length k
    overlap           = 2 # k-mer overlap
    # training
    lr                = 3e-4
    batch_size        = 32
    epochs_vae        = 5
    epochs_auxAE      = 1 # Epochs for auxiliary AE
    beta_end          = 1e-3      # final KL weight
    anneal_epochs     = 40
    # global feature
    fusion_method     = 'clustering'   # mean / weighted_mean / clustering
    n_clusters        = 5
    # paths
    output_folder     = "data" # Folder containing Train_Main.txt, natureAMP.txt
    model_root        = os.path.join(os.getcwd(), "model", "vae_nature") # Root for saving models
    aux_data_path     = "data/natureAMP.txt" # Path to auxiliary data
    pad_token         = 0 # Padding token ID

cfg = Config()
os.makedirs(cfg.output_folder, exist_ok=True)
os.makedirs(cfg.model_root, exist_ok=True)

# ────────────────────────────────────────────────────────────
# Utility : tokenizer
# ────────────────────────────────────────────────────────────
def tokenize_sequences(seq_list, k, overlap):
    """k-mer tokenization with shared dict"""
    tok_dict, tok_seqs = {}, []
    step = k - overlap # Calculate step size based on k and overlap
    if step <= 0:
        raise ValueError(f"Overlap ({overlap}) must be less than k ({k}) for tokenization step size.")

    for seq in seq_list:
        toks=[]
        # Iterate through sequence with specified step size
        for i in range(0, len(seq) - k + 1, step):
            seg = seq[i:i+k]
            if seg not in tok_dict:
                # Assign new ID if k-mer is new
                tok_dict[seg] = len(tok_dict)+1 # Start IDs from 1 (0 is pad)
            toks.append(tok_dict[seg])
        tok_seqs.append(toks)
    return tok_seqs, tok_dict

# positional enc
def positional_encoding(length, depth):
    """Generates positional encoding."""
    depth//=2
    pos  = np.arange(length)[:,None] # [length, 1]
    i    = np.arange(depth)[None,:]/depth # [1, depth/2]
    angle= pos/(10000**i) # [length, depth/2]
    pe   = np.concatenate([np.sin(angle), np.cos(angle)], axis=-1) # [length, depth]
    return tf.cast(pe, tf.float32)

# transformer encoder block
def encoder_block(x, d, heads=8, ff_mult=4, dropout=0.1):
    """Standard Transformer Encoder Block."""
    # Multi-Head Self-Attention
    attn = MultiHeadAttention(heads, key_dim=d//heads)(x,x)
    x    = LayerNormalization(epsilon=1e-6)(x + Dropout(dropout)(attn))
    # Feed-Forward Network
    ffn  = Dense(d*ff_mult, activation='relu')(x)
    ffn  = Dense(d)(ffn)
    x    = LayerNormalization(epsilon=1e-6)(x + Dropout(dropout)(ffn))
    return x

# --- Transformer Decoder Block ---
def decoder_block(inputs, context, d_model, num_heads, ff_dim, dropout=0.1, causal_mask=None, name=None):
    """Standard Transformer Decoder Block with self-attention, cross-attention, FFN."""
    # Masked Self-Attention (prevents attending to future positions)
    attn_self = MultiHeadAttention(
        num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout,
        name=f"{name}_self_attn" if name else None
    )
    attn_self_output = attn_self(
        query=inputs, value=inputs, key=inputs,
        attention_mask=causal_mask # Apply causal mask here
    )
    out1 = LayerNormalization(epsilon=1e-6, name=f"{name}_ln1" if name else None)(inputs + Dropout(dropout)(attn_self_output))

    # Cross-Attention (attends to encoder output)
    attn_cross = MultiHeadAttention(
        num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout,
        name=f"{name}_cross_attn" if name else None
    )
    attn_cross_output = attn_cross(
        query=out1, value=context, key=context # Query=decoder state, Key/Value=encoder output
    )
    out2 = LayerNormalization(epsilon=1e-6, name=f"{name}_ln2" if name else None)(out1 + Dropout(dropout)(attn_cross_output))

    # Feed-Forward Network
    ffn1 = Dense(ff_dim, activation="relu", name=f"{name}_ffn1" if name else None)(out2)
    ffn2 = Dense(d_model, name=f"{name}_ffn2" if name else None)(ffn1)
    out3 = LayerNormalization(epsilon=1e-6, name=f"{name}_ln3" if name else None)(out2 + Dropout(dropout)(ffn2))

    return out3

# --- Create Causal Mask ---
def create_causal_mask(seq_len):
    """Creates a causal mask for decoder self-attention."""
    i = tf.range(seq_len)[:, tf.newaxis]
    j = tf.range(seq_len)
    # Mask where future positions are masked (value 0)
    mask = tf.cast(i >= j, dtype="int32")
    # Add batch dimension for broadcasting: [1, seq_len, seq_len]
    return mask[tf.newaxis, :, :]

# sampling (reparameterization trick)
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian."""
    mu, logvar = args
    batch = tf.shape(mu)[0]
    dim = tf.shape(mu)[1]
    eps = tf.random.normal(shape=(batch, dim))
    return mu + tf.exp(0.5 * logvar) * eps

# KL term calculation
def kl_term(mu, logvar):
    """Calculates KL divergence between latent distribution and standard Gaussian."""
    # KL(N(mu, sigma) || N(0, 1)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # logvar = log(sigma^2)
    kl_per_sample = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mu) - tf.exp(logvar), axis=-1)
    # 确保返回的是标量，不使用 ExpandDims
    return kl_per_sample  # 返回每个样本的KL散度，形状为 [batch_size]

# mask-aware reconstruction loss
def recon_loss(y_true, y_pred):
    """Calculates sparse categorical crossentropy loss, ignoring padding."""
    # 确保使用 from_logits=False，因为 build_decoder 输出的是 softmax 概率
    ce = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=False)
    # 创建掩码以忽略填充标记
    mask = tf.cast(tf.not_equal(y_true, cfg.pad_token), tf.float32)
    # 对每个样本进行归一化处理
    ce_masked = ce * mask  # 形状: [batch_size, seq_len]
    # 计算每个样本的平均损失（忽略填充）
    # 确保除数非零
    normalizer = tf.reduce_sum(mask, axis=-1) + 1e-9  # 形状: [batch_size]
    recon_loss_per_sample = tf.reduce_sum(ce_masked, axis=-1) / normalizer  # 形状: [batch_size]
    # 返回每个样本的损失，VAE 模型会自动对批次求平均
    return tf.reduce_mean(recon_loss_per_sample)  # 返回标量，批次平均损失

# KL-β scheduler callback
class BetaAnnealer(Callback):
    """Linearly anneals the KL divergence weight (beta) during training."""
    def __init__(self, var, beta_end, anneal_epochs):
        super().__init__()
        self.v      = var # The Keras variable representing beta
        self.end    = float(beta_end) # Final beta value
        self.ep_max = int(anneal_epochs) # Number of epochs for annealing
    def on_epoch_begin(self, epoch, logs=None):
        # Calculate annealing progress (0 to 1)
        t = min(1.0, epoch / self.ep_max)
        # Update beta value
        self.v.assign(self.end * t)

# ────────────────────────────────────────────────────────────
# Data loaders
# ────────────────────────────────────────────────────────────
def load_main_sequences():
    """Loads main training sequences."""
    # Look for Train_Main.txt in the specified output_folder
    files = glob.glob(os.path.join(cfg.output_folder,"Train_Main.txt"))
    if not files:
        raise FileNotFoundError(f"Main data file 'Train_Main.txt' not found in {cfg.output_folder}")
    # Use the most recently modified file if multiple exist (though unlikely)
    f = max(files, key=os.path.getmtime)
    print("Main data:", f)
    # Read the first column assuming no header
    return pd.read_csv(f, header=None).iloc[:,0].to_numpy()

def load_aux_sequences():
    """Loads auxiliary sequences."""
    if not os.path.exists(cfg.aux_data_path):
        raise FileNotFoundError(f"Auxiliary data missing: {cfg.aux_data_path}")
    print("Aux data:", cfg.aux_data_path)
    # Read the first column assuming no header
    return pd.read_csv(cfg.aux_data_path, header=None).iloc[:,0].to_numpy()

# ────────────────────────────────────────────────────────────
# Phase-A : train Auxiliary Transformer-AE with Attention Pooling
# ────────────────────────────────────────────────────────────
def train_aux_encoder(all_tokens, vocab, aux_idx):
    """Trains auxiliary Transformer AE using Encoder-Decoder structure and Attention Pooling."""
    aux_tokens = [all_tokens[i] for i in aux_idx]
    max_len    = max(map(len, aux_tokens))
    print(f"Auxiliary AE: Max sequence length = {max_len}")
    # Pad sequences
    X_aux      = pad_sequences(aux_tokens, maxlen=max_len, padding='post', value=cfg.pad_token)

    # --- Model Input ---
    inp = Input(shape=(max_len,), dtype=tf.int32, name='ae_input')

    # --- Shared Embedding Layer ---
    # mask_zero=False because we handle padding explicitly in the loss/pooling
    embedding_layer = Embedding(vocab, cfg.embedding_dim, name='shared_embedding')

    # --- Transformer Encoder ---
    enc_emb = embedding_layer(inp)
    enc_emb += positional_encoding(max_len, cfg.embedding_dim)
    encoder_output = enc_emb
    # Stack encoder blocks
    for i in range(2): # Number of encoder layers
        # *** CORRECTED: Removed the 'name' argument ***
        encoder_output = encoder_block(encoder_output, cfg.embedding_dim, heads=8, ff_mult=4, dropout=0.1) # Removed name=...

    # --- Transformer Decoder ---
    # Decoder input is typically the target sequence shifted right (teacher forcing)
    # For AE, we can use the input sequence itself as decoder input for reconstruction
    dec_emb = embedding_layer(inp)
    dec_emb += positional_encoding(max_len, cfg.embedding_dim)

    # Create causal mask for decoder self-attention
    causal_mask = create_causal_mask(max_len)

    # Stack decoder blocks
    decoder_output = dec_emb
    for i in range(2): # Number of decoder layers
        decoder_output = decoder_block(
            decoder_output, encoder_output, # Decoder input, Encoder context
            d_model=cfg.embedding_dim,
            num_heads=8,
            ff_dim=cfg.embedding_dim * 4,
            dropout=0.1,
            causal_mask=causal_mask, # Pass the causal mask
            name=f'decoder_block_{i+1}' # Optional naming for decoder blocks
        )

    # --- Output Layer ---
    # Output logits for sparse_categorical_crossentropy with from_logits=True
    output_logits = TimeDistributed(Dense(vocab, activation=None), name='output_logits')(decoder_output)

    # --- Build Autoencoder Model ---
    ae = Model(inp, output_logits, name='AuxTransformerAE')

    # --- Masked Loss Function ---
    def masked_loss(y_true, y_pred_logits):
        y_true = tf.cast(y_true, tf.int32)
        # Calculate loss from logits
        loss = tf.keras.losses.sparse_categorical_crossentropy(
            y_true, y_pred_logits, from_logits=True
        )
        # Create mask based on padding token
        mask = tf.cast(tf.not_equal(y_true, cfg.pad_token), tf.float32)
        # Apply mask and calculate mean loss over non-padded tokens
        masked_loss_val = loss * mask
        # Add epsilon to denominator to prevent division by zero if mask sum is zero
        return tf.reduce_sum(masked_loss_val) / (tf.reduce_sum(mask) + 1e-9)

    # Add loss to the model (no need for y in fit)
    ae.add_loss(masked_loss(inp, output_logits))

    # Compile the model
    ae.compile(optimizer=Adam(cfg.lr))
    print("--- Auxiliary Transformer Autoencoder Summary ---")
    print(ae.summary())

    # --- Train Autoencoder ---
    print(f"--- Training Auxiliary AE for {cfg.epochs_auxAE} epochs ---")
    ae.fit(X_aux, # Input only, target is implicitly the input due to add_loss
           epochs=cfg.epochs_auxAE,
           batch_size=64, # Can adjust batch size
           verbose=1) # Show progress per epoch

    # --- Build Final Encoder Model with Attention Pooling ---
    # 1. Extract the trained encoder part (up to encoder_output)
    encoder_seq_model = Model(ae.input, encoder_output, name="aux_encoder_seq_output")
    encoder_seq_model.trainable = False # Freeze weights after AE training

    # 2. Apply Attention Pooling to the sequence output
    # Calculate attention scores
    attn_scores = Dense(1, activation='tanh', name='attn_pool_scores')(encoder_seq_model.output) # [B, T, 1]

    # Create mask for padding tokens in the input
    input_mask = tf.cast(tf.not_equal(encoder_seq_model.input, cfg.pad_token), dtype=tf.float32) # [B, T]
    input_mask = tf.expand_dims(input_mask, axis=-1) # [B, T, 1]

    # Mask attention scores before softmax (set padding scores to large negative)
    masked_attn_scores = attn_scores + (input_mask - 1.0) * 1e9
    # Calculate attention weights (softmax over time dimension)
    attn_weights = tf.nn.softmax(masked_attn_scores, axis=1, name='attn_pool_weights') # [B, T, 1]

    # Calculate weighted sum of encoder outputs
    # [B, T, D] * [B, T, 1] -> sum over T -> [B, D]
    pooled_enc = tf.reduce_sum(encoder_seq_model.output * attn_weights, axis=1, name='attention_pooled_enc')

    # 3. Define the final encoder model outputting the pooled vector
    enc_model = Model(encoder_seq_model.input, pooled_enc, name="aux_encoder_pooled")
    print("--- Final Pooled Encoder Model Summary ---")
    print(enc_model.summary())

    return enc_model, max_len

# ────────────────────────────────────────────────────────────
# Phase-B : Dual-Path VAE with Global Feature Enhancer
# ────────────────────────────────────────────────────────────
def build_decoder(seq_len, vocab):
    """Builds the GRU-based VAE decoder."""
    z = Input(shape=(cfg.latent_dim,), name="decoder_input_z")
    # Project latent vector and repeat
    x = Dense(cfg.hidden_dim * 2, activation='relu')(z) # Increase capacity slightly
    x = RepeatVector(seq_len)(x)
    # Stack Bidirectional GRUs
    x = Bidirectional(GRU(cfg.hidden_dim, return_sequences=True))(x)
    x = Bidirectional(GRU(cfg.hidden_dim, return_sequences=True))(x)
    # Output layer with softmax activation
    y = TimeDistributed(Dense(vocab, activation='softmax'), name="decoder_output_probs")(x)
    return Model(z, y, name="GRU_Decoder")


def build_feature_enhancer(gf):
    """Builds the global feature enhancer module."""
    n = gf.shape[0] # Number of global feature centroids
    d = gf.shape[1] # Feature dimension
    
    print(f"[DEBUG] Global feature shape: {gf.shape}, n={n}, d={d}")
    # Check if feature dimension matches latent_dim
    if d != cfg.latent_dim:
        print(f"[WARNING] Global feature dimension ({d}) does not match latent_dim ({cfg.latent_dim})")
    
    # --- Inputs ---
    z_in = Input(shape=(cfg.latent_dim,), name="enhancer_mu_in") # Original mu0
    lv_in = Input(shape=(cfg.latent_dim,), name="enhancer_lv_in") # Original lv0

    # --- Global Feature Integration ---
    if n==1:
        # If only one global feature, tile it for the batch
        gf_b = tf.tile(gf, [tf.shape(z_in)[0],1])
        print("[DEBUG] Using single global feature (tiled)")
    else:
        # Calculate similarity between input mu (z_in) and global features (gf)
        # Add explicit shape information for debugging
        print(f"[DEBUG] z_in shape: {z_in.shape}, gf shape: {gf.shape}")
        sims = tf.linalg.matmul(z_in, gf, transpose_b=True) # [B, n]
        print(f"[DEBUG] Similarity matrix shape: {sims.shape}")
        # Get weights via softmax
        w = tf.nn.softmax(sims, axis=1) # [B, n]
        # Calculate weighted average of global features
        gf_b = tf.matmul(w, gf) # [B, D]
        print(f"[DEBUG] Weighted global feature shape: {gf_b.shape}")

    # --- Enhancement Calculation Network ---
    # Concatenate original mu and the context vector (gf_b)
    concat = Concatenate()([z_in, gf_b])
    # Pass through dense layers to calculate adjustments
    x = Dense(cfg.latent_dim*2, activation='relu')(concat)
    delta_mu = Dense(cfg.latent_dim, activation='tanh', name="delta_mu")(x) # Adjustment for mu
    delta_lv = Dense(cfg.latent_dim, activation='softplus', name="delta_lv")(x) # Adjustment for lv (ensure positive)

    # --- Gated Enhancement ---
    # Learnable gate (scalar) controlling the influence of adjustments
    gamma = tf.Variable(0., trainable=True, name="gamma")
    # Calculate enhanced mu
    mu_enh = Add(name="mu_enhanced")([z_in, gamma * delta_mu])
    # Calculate enhanced lv (based on original lv_in)
    lv_enh_base = Add(name="lv_base_plus_delta")([lv_in, gamma * delta_lv])
    # Add epsilon for numerical stability
    lv_enh = Lambda(lambda t: t + 1e-6, name="lv_add_epsilon")(lv_enh_base)

    # --- Model Definition ---
    # Model takes mu0 and lv0 as input, outputs enhanced mu and lv
    return Model(inputs=[z_in, lv_in], outputs=[mu_enh, lv_enh], name='FeatureEnhancer')

def build_vae(seq_len, vocab, global_feat):
    """Builds the complete Dual-Path VAE model."""
    # --- VAE Encoder (GRU based) ---
    inp = Input(shape=(seq_len,), name='vae_input_tokens')
    # Embedding layer (mask_zero=False to avoid gradient shape issues)
    emb = Embedding(vocab, cfg.embedding_dim, mask_zero=False)(inp)
    emb += positional_encoding(seq_len, cfg.embedding_dim)
    # Stack Bidirectional GRUs
    x   = Bidirectional(GRU(cfg.hidden_dim, return_sequences=True))(emb)
    x   = Bidirectional(GRU(cfg.hidden_dim))(x) # Last GRU returns single vector
    # Dense layers to get initial mu0 and lv0
    mu0 = Dense(cfg.latent_dim, name='z_mu_raw')(x)
    lv0 = Dense(cfg.latent_dim, name='z_lv_raw')(x)

    # --- Feature Enhancer ---
    enhancer = build_feature_enhancer(global_feat)
    # Pass mu0 and lv0 to the enhancer
    mu, lv  = enhancer([mu0, lv0]) # Get enhanced mu and lv

    # --- Sampling & Decoding ---
    # Sample z from the enhanced distribution N(mu, exp(lv))
    z       = Lambda(sampling, name='z')([mu, lv])
    # Build the GRU decoder
    dec     = build_decoder(seq_len, vocab)
    # Get reconstructed output probabilities
    out     = dec(z)

    # --- VAE Model & Loss ---
    # Define the VAE model (Input: tokens, Output: reconstructed probs)
    vae = Model(inp, out, name="GlobalFeatureVAE")
    # Beta variable for KL annealing
    beta = tf.Variable(0., trainable=False, dtype=tf.float32, name='beta')
    # Calculate reconstruction loss (masked)
    recon = Lambda(lambda args: recon_loss(*args), name='recon_loss_calc')([inp, out])
    # Calculate KL divergence (mean over batch)
    kl    = Lambda(lambda args: tf.reduce_mean(kl_term(*args)), name='kl_loss_calc')([mu, lv])
    # Add combined loss (Reconstruction + beta * KL)
    vae.add_loss(recon + beta * kl)
    # Add metrics for monitoring
    vae.add_metric(recon, name='recon_loss')
    vae.add_metric(kl,    name='kl_loss')
    # 使用传统的优化器以避免 M1/M2 Mac 上的问题
    from tensorflow.keras.optimizers.legacy import Adam as LegacyAdam
    vae.compile(optimizer=LegacyAdam(cfg.lr))

    # Return VAE, decoder, beta, and the symbolic tensors mu, lv for potential later use
    return vae, dec, beta, mu, lv


# ────────────────────────────────────────────────────────────
# Main Orchestration
# ────────────────────────────────────────────────────────────
def main():
    # -- Load Data --
    print("Loading sequences...")
    main_seqs = load_main_sequences()
    aux_seqs  = load_aux_sequences()
    all_seqs  = np.concatenate([main_seqs, aux_seqs])
    print(f"Total sequences: {len(all_seqs)} (Main: {len(main_seqs)}, Aux: {len(aux_seqs)})")

    # -- Tokenization --
    print("Tokenizing sequences...")
    tok_seqs, tok_dict = tokenize_sequences(all_seqs, cfg.split_len, cfg.overlap)
    vocab = len(tok_dict) + 1 # Add 1 for padding token 0
    print(f"Vocabulary size (k={cfg.split_len}, overlap={cfg.overlap}): {vocab}")

    # -- Indices --
    aux_idx  = np.arange(len(main_seqs), len(all_seqs))
    main_idx = np.arange(len(main_seqs))

    # -- Phase-A: Train Auxiliary AE --
    print("\n--- Phase A: Training Auxiliary Encoder ---")
    aux_enc , aux_maxlen = train_aux_encoder(tok_seqs, vocab, aux_idx)
    aux_enc.trainable = False # Freeze aux encoder for feature extraction
    print("Auxiliary Encoder training complete.")

    # -- Extract Global Features --
    print("Extracting global features using auxiliary encoder...")
    X_aux = pad_sequences([tok_seqs[i] for i in aux_idx],
                          maxlen=aux_maxlen, padding='post', value=cfg.pad_token)
    feats = aux_enc.predict(X_aux, batch_size=cfg.batch_size, verbose=1) # Use configured batch_size
    print(f"Extracted features shape: {feats.shape}")

    # -- Fuse Global Features --
    if cfg.fusion_method == 'mean':
        gf = feats.mean(axis=0, keepdims=True)
        print("Global feature calculation: Mean")
    elif cfg.fusion_method == 'weighted_mean':
        # Calculate weights based on original sequence lengths (before tokenization/padding)
        aux_original_seqs = [all_seqs[i] for i in aux_idx]
        w  = np.array([len(s) for s in aux_original_seqs]); w = w / w.sum()
        gf = (feats.T @ w).T[None,...] # Weighted average
        print("Global feature calculation: Weighted Mean (by sequence length)")
    else: # clustering
        print(f"Global feature calculation: KMeans Clustering (k={cfg.n_clusters})")
        if len(feats) < cfg.n_clusters:
             print(f"Warning: Number of aux samples ({len(feats)}) is less than n_clusters ({cfg.n_clusters}). Using mean instead.")
             gf = feats.mean(axis=0, keepdims=True)
        else:
             km = KMeans(n_clusters=cfg.n_clusters, random_state=42, n_init=10).fit(feats) # Set n_init explicitly
             gf = km.cluster_centers_
    gf = tf.constant(gf, dtype=tf.float32) # Shape [n_clusters or 1, latent_dim]
    print(f"Final global features shape: {gf.shape}")

    # -- Phase-B: Train VAE --
    print("\n--- Phase B: Training VAE ---")
    main_tokens = [tok_seqs[i] for i in main_idx]
    seq_len = max(map(len, main_tokens))
    print(f"VAE: Max sequence length = {seq_len}")
    X_main  = pad_sequences(main_tokens, maxlen=seq_len, padding='post', value=cfg.pad_token)

    # Build VAE model
    vae, decoder, beta, mu_sym, lv_sym = build_vae(seq_len, vocab, gf) # Get symbolic tensors
    print("--- VAE Model Summary ---")
    print(vae.summary())

    # Create run directory
    run_dir = os.path.join(cfg.model_root,
                           f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run_dir);   print("Run-Dir:", run_dir)

    # Callbacks
    cbs = [
        BetaAnnealer(beta, cfg.beta_end, cfg.anneal_epochs),
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1), # Increased patience
        EarlyStopping(monitor='loss', patience=25, restore_best_weights=True, verbose=1),
        ModelCheckpoint(os.path.join(run_dir,'vae_best.h5'), monitor='loss', save_best_only=True, verbose=0) # Quieter checkpointing
    ]

    # Train VAE
    print(f"--- Training VAE for {cfg.epochs_vae} epochs ---")
    hist = vae.fit(X_main, # Input only
                   epochs=cfg.epochs_vae,
                   batch_size=cfg.batch_size,
                   callbacks=cbs,
                   verbose=1) # Show progress per epoch

    # --- Save Artifacts ---
    print("Saving model artifacts...")
    # Save models
    vae.save(os.path.join(run_dir,'vae_full.h5'))
    decoder.save(os.path.join(run_dir,'decoder.h5'))
    aux_enc.save(os.path.join(run_dir,'aux_encoder.h5')) # Save the final pooled aux encoder

    # Save tokenizer
    with open(os.path.join(run_dir,'tokenizer.pkl'),'wb') as f:
        pickle.dump(tok_dict, f)

    # Save hyperparameters
    with open(os.path.join(run_dir, "params.txt"), "w") as f:
        f.write(
            f"max_len={seq_len}\n" # VAE max length
            f"latent_dim={cfg.latent_dim}\n"
            f"split_times={cfg.split_len}\n" # k for k-mer
            f"overlap={cfg.overlap}\n"
            f"fusion={cfg.fusion_method}\n"
            f"n_clusters={cfg.n_clusters}\n"
            f"aux_max_len={aux_maxlen}\n" # Aux AE max length
            f"vocab_size={vocab}\n"
        )

    # Save latent samples (z) for the training data
    print("Predicting and saving z_samples for training data...")
    # Model to output z (sampled latent vector)
    z_sampler_model = Model(inputs=vae.input, outputs=vae.get_layer("z").output)
    z_samples_train = z_sampler_model.predict(X_main, batch_size=cfg.batch_size, verbose=1)
    np.save(os.path.join(run_dir, "z_samples.npy"), z_samples_train)
    print("Saved z_samples.npy")

    # Save global features used
    np.save(os.path.join(run_dir,'global_features.npy'), gf.numpy())
    print("Saved global_features.npy")


    # --- Save Training History Plot ---
    try:
        plt.figure(figsize=(8, 5))
        plt.plot(hist.history['loss'], label='Total Loss')
        if 'recon_loss' in hist.history:
             plt.plot(hist.history['recon_loss'], label='Reconstruction Loss')
        if 'kl_loss' in hist.history:
             plt.plot(hist.history['kl_loss'], label='KL Loss')
        plt.title('VAE Training Loss')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, 'loss_curve.png'))
        plt.close()
        print("Saved loss_curve.png")
    except Exception as e:
        print(f"Error saving loss curve: {e}")


    print("\nAll done ✔")

# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
