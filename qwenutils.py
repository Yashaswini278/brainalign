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

    chunk_embeddings = []
    cumulative_text = ""
    
    chunks = chunks[10:-5]   

    for i, chunk in enumerate(chunks):
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
            layers_for_this_chunk = []
            # Extract the last token embedding from each layer
            for layer_idx in range(num_layers):
                hidden_states = outputs.hidden_states[layer_idx]
                last_token_embedding = hidden_states[0, -1, :].float().cpu().numpy()
                layers_for_this_chunk.append(last_token_embedding)

            chunk_embedding = np.concatenate(layers_for_this_chunk)
            chunk_embeddings.append(chunk_embedding)
        
        # Progress indicator
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(chunks)} chunks")
    
    all_embeddings = np.stack(chunk_embeddings)
    print(f"  Final shape: {all_embeddings.shape} (chunks, layers*hidden_size)")

    return all_embeddings
