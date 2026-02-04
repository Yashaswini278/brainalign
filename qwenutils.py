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

    num_layers = model.config.num_hidden_layers
    max_length = model.config.max_position_embeddings
    print(f"  Model max context length: {max_length}")
    print(f"  Number of layers: {num_layers}")

    all_layers_embeddings = [[] for _ in range(num_layers)]
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
        inputs = tokenizer(
                        cumulative_text, 
                        return_tensors="pt", 
                        truncation=True, 
                        max_length=max_length,  
                        padding=False,
                        add_special_tokens=True
                    )
                   
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Get hidden states
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # Extract the last token embedding from each layer
            for layer_idx in range(num_layers):
                hidden_states = outputs.hidden_states[layer_idx]
                last_token_embedding = hidden_states[0, -1, :].float().cpu().numpy()
                all_layers_embeddings[layer_idx].append(last_token_embedding)
        
        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(chunks)} chunks")
    
    all_layers_embeddings = np.array([np.array(layer_embs) for layer_embs in all_layers_embeddings])
    print(f"  Final shape: {all_layers_embeddings.shape} (layers, chunks, hidden_size)")

    return all_layers_embeddings
