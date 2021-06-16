#%%
# !unzip ./dataset/data/01.Normal.zip -d /content/data
# !unzip ./dataset/data/download\ \(1\).zip -d /content/data
# %%
import json

with open('/content/data/ko_nia_normal_squad_all.json') as file:
    ai_hub_1 = json.load(file)
# %%
with open('dataset/KorQuAD_v1.0_train.json') as file:
    korquad = json.load(file)
# %%
with open('/content/data/ko_wiki_v1_squad.json') as file:
    ai_hub_2 = json.load(file)

# %%
# es_server.kill()
import os
from subprocess import Popen, PIPE, STDOUT
es_server = Popen(['/content/elasticsearch-7.9.2/bin/elasticsearch'],
                   stdout=PIPE, stderr=STDOUT,
                   preexec_fn=lambda: os.setuid(1)  # as daemon
                  )
# wait until ES has started
# ! sleep 30
# %%

# cd /content/drive/MyDrive/haystack_tutorial/
# !python squad_to_dpr.py --squad_input_filename /content/drive/MyDrive/haystack_tutorial/dataset/ko_wiki_v1_squad.json --dpr_output_filename ./dataset/aihub_dpr.json --num_hard_negative_ctxs 16
# !python squad_to_dpr.py --squad_input_filename /content/drive/MyDrive/haystack_tutorial/dataset/korquad_concat.json --dpr_output_filename ./dataset/korquad_concat_dpr.json --num_hard_negative_ctxs 16
# %%
from haystack.squad_data import SquadData
dataset = SquadData.from_file('/content/drive/MyDrive/haystack_tutorial/dataset/KorQuAD_v1.0_train_impossible.json')

#%%
dataset.merge_from_file('/content/drive/MyDrive/haystack_tutorial/dataset/ko_wiki_v1_squad_impossible.json')
dataset.save('/content/drive/MyDrive/haystack_tutorial/dataset/korquad_concat.json')
# %%
# !python squad_to_dpr.py --squad_input_filename /content/drive/MyDrive/haystack_tutorial/dataset/korquad_concat.json --dpr_output_filename ./dataset/korquad_concat_dpr.json --num_hard_negative_ctxs 16
#%%
import json
with open('/content/drive/MyDrive/haystack_tutorial/dataset/korquad_concat_dpr.json') as file:
    default = json.load(file)
#%% 
# !python squad_to_dpr.py --squad_input_filename /content/drive/MyDrive/haystack_tutorial/dataset/KorQuAD_v1.0_dev.json --dpr_output_filename ./dataset/korquad_concat_dpr_dev.json --num_hard_negative_ctxs 16