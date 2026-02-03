import numpy as np 
from pathlib import Path 
from stimulus_utils import load_grids_for_stories, load_generic_trfiles, load_simulated_trfiles
from dsutils import make_word_ds, make_phoneme_ds
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download, notebook_login

# stories = [
#     # train: 
#     ['adollshouse', 'adventuresinsayingyes', 'afatherscover', 'againstthewind', 'alternateithicatom', 'avatar', 'backsideofthestorm', 'becomingindian', 'beneaththemushroomcloud', 'birthofanation', 'bluehope', 'breakingupintheageofgoogle', 'buck', 'catfishingstrangerstofindmyself', 'cautioneating', 'christmas1940', 'cocoonoflove', 'comingofageondeathrow', 'exorcism', 'eyespy', 'firetestforlove', 'food', 'forgettingfear', 'fromboyhoodtofatherhood', 'gangstersandcookies', 'goingthelibertyway', 'goldiethegoldfish', 'golfclubbing', 'gpsformylostidentity', 'hangtime', 'haveyoumethimyet', 'howtodraw', 'ifthishaircouldtalk', 'inamoment', 'itsabox', 'jugglingandjesus', 'kiksuya', 'leavingbaghdad', 'legacy', 'life', 'lifeanddeathontheoregontrail', 'lifereimagined', 'listo', 'mayorofthefreaks', 'metsmagic', 'mybackseatviewofagreatromance', 'myfathershands', 'myfirstdaywiththeyankees', 'naked', 'notontheusualtour', 'odetostepfather', 'onlyonewaytofindout', 'penpal', 'quietfire', 'reachingoutbetweenthebars', 'shoppinginchina', 'singlewomanseekingmanwich', 'sloth', 'souls', 'stagefright', 'stumblinginthedark', 'superheroesjustforeachother', 'sweetaspie', 'swimmingwithastronauts', 'thatthingonmyarm', 'theadvancedbeginner', 'theclosetthatateeverything', 'thecurse', 'thefreedomridersandme', 'theinterview', 'thepostmanalwayscalls', 'theshower', 'thetiniestbouquet', 'thetriangleshirtwaistconnection', 'threemonths', 'thumbsup', 'tildeath', 'treasureisland', 'undertheinfluence', 'vixenandtheussr', 'waitingtogo', 'whenmothersbullyback', 'wildwomenanddancingqueens'],

#     # test:
#     ['wheretheressmoke']
# ]

model_name = "Qwen/Qwen3-4B" # 36 layers ; layer_dim = 2560 

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto"
)
model.eval()

story = ['adollshouse']



