#%%
from farm.modeling.language_model import LanguageModel
from haystack.pipeline import Pipeline
#################################################
#%%
from haystack.retriever.dense import DensePassageRetriever
from haystack.document_store.memory import InMemoryDocumentStore

doc_dir = "dataset/dpr_training/"

train_filename = "train/korquad_dpr_train.json"
dev_filename = "dev/korquad_dpr_dev.json"

query_model = "kykim/bert-kor-base"
passage_model = "kykim/bert-kor-base"

save_dir = "saved_models/dpr"

# %%
retriever = DensePassageRetriever(
    document_store=InMemoryDocumentStore(),
    query_embedding_model=query_model,
    passage_embedding_model=passage_model,
    max_seq_len_query=64,
    max_seq_len_passage=256
)
#%%
retriever.train(
    data_dir=doc_dir,
    train_filename=train_filename,
    dev_filename=dev_filename,
    test_filename=dev_filename,
    n_epochs=1,
    batch_size=2,
    grad_acc_steps=8,
    save_dir=save_dir,
    evaluate_every=3000,
    embed_title=True,
    num_positives=1,
    num_hard_negatives=3,
    # use_amp="O2"
)