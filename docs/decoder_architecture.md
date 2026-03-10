# Qwen3TTSTokenizerV2Decoder Architecture

> Config: [Qwen3-TTS-12Hz-0.6B-Base/speech_tokenizer/config.json](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base/raw/main/speech_tokenizer/config.json)

## Overall Forward Flow

```mermaid
flowchart TD
    Input["codes<br/>[B, 16, T]"]

    subgraph Quantizer["SplitResidualVectorQuantizer.decode"]
        Q1["rvq_first (semantic, n_q=1)<br/>Conv1d(512→256) → Codebook(256, 2048) → Conv1d(256→512)"]
        Q2["rvq_rest (acoustic, n_q=15)<br/>Conv1d(512→256) → Codebook(256, 2048)×15 → Conv1d(256→512)"]
        QSum(["rvq_first + rvq_rest"])
        Q1 --> QSum
        Q2 --> QSum
    end

    PreConv["pre_conv: CausalConvNet(512→1024, k=3)"]
    Transpose1["transpose(1,2)<br/>[B, 1024, T] → [B, T, 1024]"]

    subgraph Transformer["pre_transformer: TransformerModel"]
        TIn["input_proj: Linear(1024→512)"]
        TLayers["TransformerLayer × 8<br/>(sliding_window=72, RoPE)"]
        TNorm["norm: RMSNorm(512)"]
        TOut["output_proj: Linear(512→1024)"]
        TIn --> TLayers --> TNorm --> TOut
    end

    Permute["permute(0,2,1)<br/>[B, T, 1024] → [B, 1024, T]"]

    subgraph Upsample["upsample (upsampling_ratios=[2, 2])"]
        U0["TransConvNet(1024→1024, k=2, s=2) + ConvNeXtBlock(1024)<br/>×2 → [B, 1024, T×2]"]
        U1["TransConvNet(1024→1024, k=2, s=2) + ConvNeXtBlock(1024)<br/>×2 → [B, 1024, T×4]"]
        U0 --> U1
    end

    subgraph Decoder["decoder (upsample_rates=[8, 5, 4, 3])"]
        DPre["CausalConvNet(1024→1536, k=7)"]
        D0["DecoderBlock[0]: ×8<br/>1536→768 → [B, 768, T×32]"]
        D1["DecoderBlock[1]: ×5<br/>768→384 → [B, 384, T×160]"]
        D2["DecoderBlock[2]: ×4<br/>384→192 → [B, 192, T×640]"]
        D3["DecoderBlock[3]: ×3<br/>192→96 → [B, 96, T×1920]"]
        DSnake["SnakeBeta(96)"]
        DOut["CausalConvNet(96→1, k=7)"]
        DPre --> D0 --> D1 --> D2 --> D3 --> DSnake --> DOut
    end

    Clamp["clamp(-1, 1)"]
    Output["waveform<br/>[B, 1, T×1920]"]

    Input --> Quantizer
    QSum --> PreConv --> Transpose1 --> Transformer
    TOut --> Permute --> Upsample
    U1 --> Decoder
    DOut --> Clamp --> Output

    style Input fill:#e3f2fd,stroke:#1565c0
    style Output fill:#e8f5e9,stroke:#2e7d32
    style Quantizer fill:#fff3e0,stroke:#e65100
    style Transformer fill:#f3e5f5,stroke:#6a1b9a
    style Upsample fill:#e0f7fa,stroke:#00695c
    style Decoder fill:#fce4ec,stroke:#b71c1c
```

## TransformerLayer Detail

```mermaid
flowchart TD
    subgraph Layer["TransformerLayer (×8, sliding_window=72)"]
        LIn["hidden_states<br/>[B, T, 512]"]

        subgraph AttnBlock["Self-Attention Block"]
            LN1["input_layernorm: RMSNorm(512, ε=1e-5)"]
            Attn["Attention<br/>q_proj: Linear(512→1024) [16 heads, dim=64]<br/>k_proj: Linear(512→1024) [16 KV heads]<br/>v_proj: Linear(512→1024)<br/>+ RoPE (θ=10000)<br/>o_proj: Linear(1024→512)"]
            AScale["self_attn_layer_scale (init=0.01)"]
            LN1 --> Attn --> AScale
        end

        ARes(["+ residual"])

        subgraph MLPBlock["MLP Block"]
            LN2["post_attention_layernorm: RMSNorm(512, ε=1e-5)"]
            MLP["SwiGLU MLP<br/>gate_proj: Linear(512→1024)<br/>up_proj: Linear(512→1024)<br/>act_fn: SiLU<br/>down_proj: Linear(1024→512)"]
            MScale["mlp_layer_scale (init=0.01)"]
            LN2 --> MLP --> MScale
        end

        MRes(["+ residual"])
        LOut["hidden_states<br/>[B, T, 512]"]

        LIn --> AttnBlock
        LIn --> ARes
        AScale --> ARes
        ARes --> MLPBlock
        ARes --> MRes
        MScale --> MRes
        MRes --> LOut
    end

    style AttnBlock fill:#f3e5f5,stroke:#6a1b9a
    style MLPBlock fill:#ede7f6,stroke:#4527a0
```

## DecoderBlock Detail

```mermaid
flowchart TD
    subgraph DBlock["DecoderBlock[i]"]
        DIn["input<br/>[B, in_dim, L]"]
        Snake1["SnakeBeta(in_dim)"]
        TConv["CausalTransConvNet(in_dim→out_dim, k=2×rate, stride=rate)"]

        subgraph RU1["ResidualUnit (dilation=1)"]
            R1S1["SnakeBeta(out_dim)"]
            R1C1["CausalConvNet(out_dim→out_dim, k=7, d=1)"]
            R1S2["SnakeBeta(out_dim)"]
            R1C2["CausalConvNet(out_dim→out_dim, k=1)"]
            R1Res(["+ residual"])
            R1S1 --> R1C1 --> R1S2 --> R1C2 --> R1Res
        end

        subgraph RU2["ResidualUnit (dilation=3)"]
            R2S1["SnakeBeta(out_dim)"]
            R2C1["CausalConvNet(out_dim→out_dim, k=7, d=3)"]
            R2S2["SnakeBeta(out_dim)"]
            R2C2["CausalConvNet(out_dim→out_dim, k=1)"]
            R2Res(["+ residual"])
            R2S1 --> R2C1 --> R2S2 --> R2C2 --> R2Res
        end

        subgraph RU3["ResidualUnit (dilation=9)"]
            R3S1["SnakeBeta(out_dim)"]
            R3C1["CausalConvNet(out_dim→out_dim, k=7, d=9)"]
            R3S2["SnakeBeta(out_dim)"]
            R3C2["CausalConvNet(out_dim→out_dim, k=1)"]
            R3Res(["+ residual"])
            R3S1 --> R3C1 --> R3S2 --> R3C2 --> R3Res
        end

        DOut["output<br/>[B, out_dim, L×rate]"]

        DIn --> Snake1 --> TConv --> RU1
        R1Res --> RU2
        R2Res --> RU3
        R3Res --> DOut
    end

    style DBlock fill:#fce4ec,stroke:#b71c1c
    style RU1 fill:#ffebee,stroke:#c62828
    style RU2 fill:#ffebee,stroke:#c62828
    style RU3 fill:#ffebee,stroke:#c62828
```

## DecoderBlock Parameters

| Block | upsample_rate | in_dim | out_dim | TransConv kernel | Output size |
|-------|:---:|:---:|:---:|:---:|:---:|
| DecoderBlock[0] | 8 | 1536 | 768 | 16 | [B, 768, T×32] |
| DecoderBlock[1] | 5 | 768 | 384 | 10 | [B, 384, T×160] |
| DecoderBlock[2] | 4 | 384 | 192 | 8 | [B, 192, T×640] |
| DecoderBlock[3] | 3 | 192 | 96 | 6 | [B, 96, T×1920] |

> `in_dim = decoder_dim // 2^i`, `out_dim = decoder_dim // 2^(i+1)`, `kernel = 2 × rate`

## ConvNeXtBlock Detail

```mermaid
flowchart TD
    subgraph CNB["ConvNeXtBlock(1024)"]
        CIn["input<br/>[B, 1024, L]"]
        DW["dwconv: CausalConvNet(1024→1024, k=7, groups=1024)<br/>depthwise convolution"]
        Perm1["permute(0,2,1) → [B, L, 1024]"]
        LN["LayerNorm(1024, ε=1e-6)"]
        PW1["pwconv1: Linear(1024→4096)"]
        Act["GELU"]
        PW2["pwconv2: Linear(4096→1024)"]
        Gamma["× γ (learnable, init=1e-6)"]
        Perm2["permute(0,2,1) → [B, 1024, L]"]
        Res(["+ residual"])
        COut["output<br/>[B, 1024, L]"]

        CIn --> DW --> Perm1 --> LN --> PW1 --> Act --> PW2 --> Gamma --> Perm2 --> Res --> COut
        CIn --> Res
    end

    style CNB fill:#e0f7fa,stroke:#00695c
```

## Upsampling Calculation

```
Token rate: 12.5 Hz (12.5 tokens per second)

upsampling_ratios: 2 × 2 = 4×
upsample_rates:    8 × 5 × 4 × 3 = 480×
────────────────────────────────────────
Total: 4 × 480 = 1,920×

Output sample rate: 12.5 Hz × 1,920 = 24,000 Hz (24 kHz)
```

## SnakeBeta Activation Function

```
SnakeBeta(x) = x + (1/β) × sin²(αx)

  α, β: per-channel learnable parameters (exponentiated to ensure positivity)
  I/O:  [B, C, T] → [B, C, T]
```

## Class Hierarchy

```mermaid
classDiagram
    class Qwen3TTSTokenizerV2Decoder {
        +quantizer: SplitResidualVectorQuantizer
        +pre_conv: CausalConvNet
        +pre_transformer: TransformerModel
        +upsample: ModuleList
        +decoder: ModuleList
        +forward(codes) → waveform
        +chunked_decode(codes, chunk_size, left_context_size)
    }

    class SplitResidualVectorQuantizer {
        +rvq_first: ResidualVectorQuantizer (n_q=1)
        +rvq_rest: ResidualVectorQuantizer (n_q=15)
        +decode(codes) → [B, 512, T]
    }

    class ResidualVectorQuantizer {
        +input_proj: Conv1d / Identity
        +output_proj: Conv1d / Identity
        +vq: ResidualVectorQuantization
        +decode(codes) → quantized
    }

    class ResidualVectorQuantization {
        +layers: ModuleList~VectorQuantization~
        +decode(codes) → quantized
    }

    class VectorQuantization {
        +_codebook: EuclideanCodebook
        +project_out: Linear / Identity
        +decode(codes) → quantized
    }

    class EuclideanCodebook {
        +dim: 256
        +codebook_size: 2048
        +cluster_usage: Parameter
        +embedding_sum: Parameter
        +decode(codes) → embeddings
    }

    class TransformerModel {
        +input_proj: Linear(1024→512)
        +layers: ModuleList~TransformerLayer~ ×8
        +norm: RMSNorm(512)
        +output_proj: Linear(512→1024)
        +rotary_emb: RotaryEmbedding
    }

    class TransformerLayer {
        +input_layernorm: RMSNorm
        +self_attn: Attention
        +self_attn_layer_scale: LayerScale
        +post_attention_layernorm: RMSNorm
        +mlp: SwiGLU MLP
        +mlp_layer_scale: LayerScale
    }

    class DecoderBlock {
        +SnakeBeta → TransConvNet → ResidualUnit ×3
    }

    Qwen3TTSTokenizerV2Decoder --> SplitResidualVectorQuantizer
    Qwen3TTSTokenizerV2Decoder --> TransformerModel
    Qwen3TTSTokenizerV2Decoder --> DecoderBlock
    SplitResidualVectorQuantizer --> ResidualVectorQuantizer
    ResidualVectorQuantizer --> ResidualVectorQuantization
    ResidualVectorQuantization --> VectorQuantization
    VectorQuantization --> EuclideanCodebook
    TransformerModel --> TransformerLayer
```

## Config Parameters

### decoder_config

| Parameter | Value | Description |
|-----------|-------|-------------|
| `codebook_size` | 2048 | Number of codebook entries |
| `codebook_dim` | 512 | Codebook dimension |
| `hidden_size` | 512 | Transformer hidden dimension |
| `latent_dim` | 1024 | Latent representation dimension |
| `num_hidden_layers` | 8 | Number of Transformer layers |
| `num_attention_heads` | 16 | Number of attention heads |
| `num_key_value_heads` | 16 | Number of KV heads |
| `head_dim` | 64 | Dimension per head |
| `intermediate_size` | 1024 | MLP intermediate dimension |
| `hidden_act` | silu | MLP activation function |
| `sliding_window` | 72 | Sliding window size |
| `max_position_embeddings` | 8000 | Maximum position embedding length |
| `rope_theta` | 10000 | RoPE base period |
| `layer_scale_initial_scale` | 0.01 | Initial value for LayerScale |
| `rms_norm_eps` | 1e-5 | ε for RMSNorm |
| `num_quantizers` | 16 | Number of quantizers |
| `decoder_dim` | 1536 | Initial decoder channel count |
| `upsample_rates` | [8, 5, 4, 3] | Upsample rates for waveform generation stages |
| `upsampling_ratios` | [2, 2] | Upsample ratios after Transformer |
