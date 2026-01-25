import torch
from model import build_transformer
from config import *
import tiktoken

def generate_text(prompt, model, tokenizer, max_length=200, temperature=0.8, device='cuda'):
    model.eval()
    
    # Encode prompt
    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    
    # Generate
    with torch.no_grad():
        for _ in range(max_length):
            # Get logits for last token
            mask = torch.tril(torch.ones(tokens.shape[1], tokens.shape[1])).unsqueeze(0).unsqueeze(0).to(device)
            logits = model(tokens, mask)
            logits = logits[:, -1, :] / temperature
            
            # Sample next token
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append and check for end
            tokens = torch.cat([tokens, next_token], dim=1)
            if next_token.item() == tokenizer.eot_token:
                break
    
    # Decode
    generated = tokenizer.decode(tokens[0].cpu().numpy())
    return generated

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load tokenizer
    tokenizer = tiktoken.encoding_for_model("gpt-4")
    vocab_size = tokenizer.n_vocab

    map_location = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    print("Loading model...")
    model = build_transformer(vocab_size, D_MODEL, H, N, D_FF, DROPOUT).to(device)
    model.load_state_dict(torch.load('weights/best_model.pt', map_location=map_location))
    print("Model loaded!")
    
    # Test prompts
    prompts = [
        "Once upon a time, there was a little",
        "The brave knight went to the",
        "In a magical forest, a small"
    ]
    
    print("\n" + "="*60)
    print("GENERATING STORIES")
    print("="*60)
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n[Story {i}] Prompt: {prompt}")
        print("-" * 60)
        generated = generate_text(prompt, model, tokenizer, max_length=150, device=device)
        print(generated)
        print()