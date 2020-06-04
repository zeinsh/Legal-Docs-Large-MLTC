import numpy as np

# transformers
from transformers import PreTrainedModel, PreTrainedTokenizer, PretrainedConfig

from transformers import BertForSequenceClassification, BertTokenizer, BertConfig
from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig
from transformers import XLNetForSequenceClassification, XLNetTokenizer, XLNetConfig
from transformers import XLMForSequenceClassification, XLMTokenizer, XLMConfig
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, DistilBertConfig
# from transformers import AlbertForSequenceClassification, AlbertTokenizer, AlbertConfig

from fastai.text.transform import BaseTokenizer
from fastai.callbacks.csv_logger import CSVLogger
from fastai.text import List, Vocab, Collection, nn, Tokenizer, NumericalizeProcessor, TokenizeProcessor

from fastai.callbacks import *
from fastai.basic_train import Learner
from fastai.train import ShowGraph
from transformers import AdamW
from functools import partial

from fastai.metrics import accuracy_thresh, fbeta
from evaluation import multi_label_precision, multi_label_recall

import torch

MODEL_CLASSES = {
    'bert': (BertForSequenceClassification, BertTokenizer, BertConfig),
    'xlnet': (XLNetForSequenceClassification, XLNetTokenizer, XLNetConfig),
    'xlm': (XLMForSequenceClassification, XLMTokenizer, XLMConfig),
    'roberta': (RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig),
    'distilbert': (DistilBertForSequenceClassification, DistilBertTokenizer, DistilBertConfig),
    #'albert':(AlbertForSequenceClassification, AlbertTokenizer, AlbertConfig)
    }


def getTrainingMetrics():
    thresh=0.5
    acc_05 = partial(accuracy_thresh, thresh=thresh)
    prc_05 = partial(multi_label_precision, thresh=thresh)
    rec_05 = partial(multi_label_recall, thresh=thresh)
    f_05   = partial(fbeta, thresh=thresh, beta=1)
    
    return [acc_05, prc_05, rec_05, f_05]

class TransformersBaseTokenizer(BaseTokenizer):
    """Wrapper around PreTrainedTokenizer to be compatible with fast.ai"""
    def __init__(self, pretrained_tokenizer: PreTrainedTokenizer, model_type = 'bert', maxlen=100000, **kwargs):
        self._pretrained_tokenizer = pretrained_tokenizer
        self.max_seq_len = min(pretrained_tokenizer.max_len, maxlen)
        self.model_type = model_type
        
    def __call__(self, *args, **kwargs): 
        return self

    def tokenizer(self, t:str) -> List[str]:
        """Limits the maximum sequence length and add the spesial tokens"""
        CLS = self._pretrained_tokenizer.cls_token
        SEP = self._pretrained_tokenizer.sep_token
        if self.model_type in ['roberta']:
            tokens = self._pretrained_tokenizer.tokenize(t, add_prefix_space=True)[:self.max_seq_len - 2]
        else:
            tokens = self._pretrained_tokenizer.tokenize(t)[:self.max_seq_len - 2]
        return [CLS] + tokens + [SEP]

class TransformersVocab(Vocab):
    def __init__(self, tokenizer: PreTrainedTokenizer):
        super(TransformersVocab, self).__init__(itos = [])
        self.tokenizer = tokenizer
    
    def numericalize(self, t:Collection[str]) -> List[int]:
        "Convert a list of tokens `t` to their ids."
        return self.tokenizer.convert_tokens_to_ids(t)
        #return self.tokenizer.encode(t)

    def textify(self, nums:Collection[int], sep=' ') -> List[str]:
        "Convert a list of `nums` to their tokens."
        nums = np.array(nums).tolist()
        return sep.join(self.tokenizer.convert_ids_to_tokens(nums)) if sep is not None else self.tokenizer.convert_ids_to_tokens(nums)

# defining our model architecture 
class CustomTransformerModel(nn.Module):
    def __init__(self, transformer_model: PreTrainedModel):
        super(CustomTransformerModel,self).__init__()
        self.transformer = transformer_model
        
    def forward(self, input_ids, attention_mask=None):
        
        #attention_mask = (input_ids!=1).type(input_ids.type()) # Test attention_mask for RoBERTa
        
        logits = self.transformer(input_ids,
                                attention_mask = attention_mask)[0]   
        return logits

def getTransformerProcecssor(tokenizer_class, pretrained_model_name, model_type, maxlen=512):
    transformer_tokenizer = tokenizer_class.from_pretrained(pretrained_model_name)
    transformer_base_tokenizer = TransformersBaseTokenizer(pretrained_tokenizer = transformer_tokenizer, model_type = model_type, maxlen=maxlen)
    fastai_tokenizer = Tokenizer(tok_func = transformer_base_tokenizer, pre_rules=[], post_rules=[])
    
    transformer_vocab =  TransformersVocab(tokenizer = transformer_tokenizer)
    numericalize_processor = NumericalizeProcessor(vocab=transformer_vocab)

    tokenize_processor = TokenizeProcessor(tokenizer=fastai_tokenizer, include_bos=False, include_eos=False)
    return [tokenize_processor, numericalize_processor]

def getListLayersBert(learner):
    # For bert-uncased-base
    list_layers_bert = [learner.model.transformer.bert.embeddings,
                  learner.model.transformer.bert.encoder.layer[0],
                  learner.model.transformer.bert.encoder.layer[1],
                  learner.model.transformer.bert.encoder.layer[2],
                  learner.model.transformer.bert.encoder.layer[3],
                  learner.model.transformer.bert.encoder.layer[4],
                  learner.model.transformer.bert.encoder.layer[5],
                  learner.model.transformer.bert.encoder.layer[6],
                  learner.model.transformer.bert.encoder.layer[7],
                  learner.model.transformer.bert.encoder.layer[8],
                  learner.model.transformer.bert.encoder.layer[9],
                  learner.model.transformer.bert.encoder.layer[10],
                  learner.model.transformer.bert.encoder.layer[11],
                  learner.model.transformer.bert.pooler]
    
    return list_layers_bert

def getListLayersRoberta(learner):
    # For bert-uncased-base
    list_layers_bert = [learner.model.transformer.roberta.embeddings,
                  learner.model.transformer.roberta.encoder.layer[0],
                  learner.model.transformer.roberta.encoder.layer[1],
                  learner.model.transformer.roberta.encoder.layer[2],
                  learner.model.transformer.roberta.encoder.layer[3],
                  learner.model.transformer.roberta.encoder.layer[4],
                  learner.model.transformer.roberta.encoder.layer[5],
                  learner.model.transformer.roberta.encoder.layer[6],
                  learner.model.transformer.roberta.encoder.layer[7],
                  learner.model.transformer.roberta.encoder.layer[8],
                  learner.model.transformer.roberta.encoder.layer[9],
                  learner.model.transformer.roberta.encoder.layer[10],
                  learner.model.transformer.roberta.encoder.layer[11],
                  learner.model.transformer.roberta.pooler]
    
    return list_layers_bert

def getListLayerXLNet(learner):
    list_layers = [learner.model.transformer.transformer.word_embedding,
              learner.model.transformer.transformer.layer[0],
              learner.model.transformer.transformer.layer[1],
              learner.model.transformer.transformer.layer[2],
              learner.model.transformer.transformer.layer[3],
              learner.model.transformer.transformer.layer[4],
              learner.model.transformer.transformer.layer[5],
              learner.model.transformer.transformer.layer[6],
              learner.model.transformer.transformer.layer[7],
              learner.model.transformer.transformer.layer[8],
              learner.model.transformer.transformer.layer[9],
              learner.model.transformer.transformer.layer[10],
              learner.model.transformer.transformer.layer[11],
              learner.model.transformer.sequence_summary]

    return list_layers
def getListLayerDistilbert(learner):
    list_layers = [learner.model.transformer.distilbert.embeddings,
                learner.model.transformer.distilbert.transformer.layer[0],
                learner.model.transformer.distilbert.transformer.layer[1],
                learner.model.transformer.distilbert.transformer.layer[2],
                learner.model.transformer.distilbert.transformer.layer[3],
                learner.model.transformer.distilbert.transformer.layer[4],
                learner.model.transformer.distilbert.transformer.layer[5],
                  ]
    return list_layers

def getListLayerAlbert(learner):
    list_layers = [learner.model.transformer.albert.embeddings,
                   learner.model.transformer.albert.encoder,
                  ]
    return list_layers

def getListLayers(learner, model_type='bert'):
    if model_type=='roberta':
        return getListLayersRoberta(learner)
    elif model_type=='xlnet':
        return  getListLayerXLNet(learner)
    elif model_type=='distilbert':
        return getListLayerDistilbert(learner)
    #elif model_type=='albert':
    #    return getListLayerAlbert(learner)
    else:
        return getListLayersBert(learner) #Bert is default


def getLearner(data_clas, pretrained_model_name, model_class, config_class, use_fp16, logfilename='history', append=False, model_type='bert'):

    config = config_class.from_pretrained(pretrained_model_name)
    config.num_labels = data_clas.train_dl.c
    config.use_bfloat16 = use_fp16

    transformer_model = model_class.from_pretrained(pretrained_model_name, config = config)
    custom_transformer_model = CustomTransformerModel(transformer_model = transformer_model)

    CustomAdamW = partial(AdamW, correct_bias=False)

    learner = Learner(data_clas, 
                  custom_transformer_model, 
                  opt_func = CustomAdamW, 
                  metrics=getTrainingMetrics(), 
                      callback_fns=[partial(CSVLogger, filename=logfilename, append=append)])


    # Show graph of learner stats and metrics after each epoch.
    learner.callbacks.append(ShowGraph(learner))

    # Put learn in FP16 precision mode. --> Seems to not working
    if use_fp16: learner = learner.to_fp16()
        
    list_layers_bert=getListLayers(learner, model_type=model_type)
    learner.split(list_layers_bert)
    
    return learner