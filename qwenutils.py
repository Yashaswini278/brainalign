import torch 
import numpy as np 

def embed_text_qwen_all_layers(chunks, model, tokenizer, batch_size=1):
    """
    Extract embeddings from all layers of Qwen3 model for given text chunks.
    Each chunk is processed with cumulative context from previous chunks.
    
    Args:
        chunks: List of text strings (one per TR)
        model: Qwen3-4B model
        tokenizer: Qwen3-4B tokenizer
        batch_size: Number of chunks to process at once
    
    Returns:
        numpy array of shape (n_layers, n_chunks, hidden_size)
    """

    num_layers = 36
    all_layers_embeddings = []
    
    for layer_idx in range(num_layers):
        layer_embeddings = []
        cumulative_text = ""
        
        for i, chunk in enumerate(chunks):
            # Skip empty chunks
            if not chunk or not chunk.strip():
                continue
            # Add current chunk to cumulative context
            if cumulative_text:
                cumulative_text += " " + chunk
            else:
                cumulative_text = chunk
            
            # Tokenize the cumulative text
            inputs = tokenizer(cumulative_text, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Get hidden states
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[layer_idx]  # Shape: (batch, seq_len, hidden_size)
                
                # Get the last token's embedding (most recent context)
                last_token_embedding = hidden_states[0, -1, :].cpu().numpy()
                layer_embeddings.append(last_token_embedding)
        
        all_layers_embeddings.append(np.array(layer_embeddings))
        print(f"  Completed layer {layer_idx}/{num_layers-1}")
    
    return np.array(all_layers_embeddings)
