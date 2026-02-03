import numpy as np 
import json 
import os 
from pathlib import Path 
from stimulus_utils import load_grids_for_stories, load_generic_trfiles, load_simulated_trfiles
from dsutils import make_word_ds, make_phoneme_ds
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download, notebook_login
from qwenutils import embed_text_qwen_all_layers
import argparse

parser = argparse.ArgumentParser(description="Extract Qwen3 embeddings for stories from all layers.")
parser.add_argument("--base_dir", type=str, default=None,
                    help="Base directory path (default: current working directory)")
parser.add_argument("--grid_path", type=str, default="new_stories_data/stimuli/TextGrids",
                    help="Path to TextGrids directory relative to base_dir (default: new_stories_data/stimuli/TextGrids)")
parser.add_argument("--respdict_path", type=str, default="new_stories_data/stimuli/respdict.json",
                    help="Path to respdict.json relative to base_dir (default: new_stories_data/stimuli/respdict.json)")
parser.add_argument("--output_dir", type=str, default="new_stories_data/features",
                    help="Output directory for embeddings relative to base_dir (default: new_stories_data/features)")
args = parser.parse_args()

stories = ['adollshouse']

# Set up paths
base_dir = Path(args.base_dir) if args.base_dir else Path.cwd()
grid_path = args.grid_path
respdict_path = args.respdict_path
output_dir = base_dir / args.output_dir

with open(base_dir / respdict_path, "r") as f:
    respdict = json.load(f)

grid_dir = base_dir / grid_path

# Load TextGrids
grids = load_grids_for_stories(stories, grid_dir)

# Load TRfiles
trfiles = load_simulated_trfiles(respdict)

# Make word and phoneme datasequences
wordseqs = make_word_ds(grids, trfiles)
phonseqs = make_phoneme_ds(grids, trfiles)

model_name = "Qwen/Qwen3-4B" # 36 layers ; layer_dim = 2560 

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto"
)
model.eval()

for story_name in stories:
    print(f"\nProcessing {story_name}...")
    wordseq = wordseqs[story_name]
    
    # Each chunk = words during one TR 
    chunks = [" ".join(words) for words in wordseq.chunks()]
    print(f"  Number of chunks: {len(chunks)}")
    
    # Embed each TR's words with cumulative context from all layers
    embeddings_all_layers = embed_text_qwen_all_layers(chunks, model, tokenizer, batch_size=1)
    
    print(f"  {story_name}: {embeddings_all_layers.shape} (layers, chunks, hidden_size)")
    
    # Save embeddings for this story (all layers) to a separate file
    output_file = output_dir / f'stories-qwen3-4b-features_all_layers_{story_name}.npz'

print(f"\n✓ All stories processed!")
print(f"Total stories: {len(stories)}")


