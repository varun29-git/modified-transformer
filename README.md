```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'fontSize': '14px', 'fontFamily': 'arial', 'primaryColor': '#fff', 'edgeLabelBackground':'#fff', 'tertiaryColor': '#f4f4f4'}}}%%
flowchart TD
    %% Nodes
    In([Inputs]) --> Emb[Input Embedding]
    
    subgraph Decoder ["Decoder Stack (Nx Layers)"]
        direction TB
        
        %% Block Input
        Emb --> BlockIn(( ))
        
        %% Sublayer 1: MHA
        BlockIn -->|Pre-Norm| Norm1[RMSNorm]
        Norm1 --> MHA["Masked Multi-Head Attn<br/>(Rotary Embeddings)"]
        MHA --> Drop1[Dropout]
        
        %% Residual 1
        Drop1 --> Add1((+))
        BlockIn -->|Residual| Add1
        
        %% Sublayer 2: FFN
        Add1 -->|Pre-Norm| Norm2[RMSNorm]
        Norm2 --> FFN["Feed Forward<br/>SwiGLU / SiLU"]
        FFN --> Drop2[Dropout]
        
        %% Residual 2
        Drop2 --> Add2((+))
        Add1 -->|Residual| Add2
    end

    %% Final Output
    Add2 --> FinalNorm[Final RMSNorm]
    FinalNorm --> Linear[Linear Projection]
    Linear --> Softmax[Log Softmax]
    Softmax --> Out([Output Probabilities])

    %% Styling to mimic paper blocks
    classDef box fill:#fff,stroke:#333,stroke-width:1px;
    classDef gray fill:#f5f5f5,stroke:#333,stroke-width:1px;
    classDef circle width:0px,height:0px,stroke-width:0px;
    
    class Emb,MHA,FFN,Linear,Softmax box;
    class Norm1,Norm2,FinalNorm,Drop1,Drop2 gray;
    class BlockIn circle;
