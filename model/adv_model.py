import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.modeling_outputs import SequenceClassifierOutput

import pdb
import numpy as np

from scipy.stats import binomtest

# calculate length of each sample in batch
def batch_len(input_ids, pad_idx=0):
    b_length = (input_ids != pad_idx).data.sum(dim=-1)
    return b_length

class SeqClsWrapper(nn.Module):
    def __init__(self, model, args):
        super(SeqClsWrapper, self).__init__()
        self.model = model
        self.num_classes = args["num_classes"]
        self.device = args["device"]
        self.pooler_output = args["pooler_output"]
        self.transformer = args["model"]
        self.custom_forward = args["custom_forward"]

        self.pad_idx = args["pad_idx"]
        self.mask_idx = args["mask_idx"]
        self.multi_mask = args["multi_mask"]

        self.vote_type = args["vote_type"]
        self.binom_p = args["binom_p"]
        self.alpha_p = args["alpha_p"]

        self.binom_ensemble = args["binom_ensemble"]
        self.num_ensemble = args["num_ensemble"]
        self.w_gn = args["w_gn"]
        self.binom_n_eval = args["binom_n_eval"]


    def forward(self, input_ids, attention_mask, labels=None, delta_grad=None, grad_idx=None):
        """
        - custom_forward: to add noise to word embeddings
        """
        if self.custom_forward:
            if delta_grad is not None:
                word_embeds = self.get_word_embeddings(input_ids)
                emb_grad = word_embeds + self.w_gn * delta_grad # Robust embedding: noise to word embedding (section 2.2)
                outputs = self.transformer_forward(inputs_embeds=emb_grad, attention_mask=attention_mask)
                output = self.classifier_forward(outputs, labels, grad_idx)
            else:
                outputs = self.transformer_forward(input_ids=input_ids, attention_mask=attention_mask)
                output = self.classifier_forward(outputs, labels, grad_idx)
        else:
            output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return output

    def transformer_forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = self.model.config.return_dict
        encoder = getattr(self.model, self.transformer)

        outputs = encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return outputs

    def classifier_forward(self, outputs, labels=None, grad_idx=None):
        """
        - outputs[0]: last_hidden_state, bs x seq x dim
        - outputs[1]: pooler output, bs x dim
        """
        return_dict = self.model.config.return_dict

        if grad_idx is not None:
            sent_idx = [idx[0] for idx in grad_idx]

        if self.transformer=='bert':
            if self.pooler_output:
                pooled_output = outputs[1]
                pooled_output = self.dropout(pooled_output)
                logits = self.model.classifier(pooled_output)
            else:
                sequence_output = outputs[0] # (bs, seq_len, dim)
                seq_inp = sequence_output[:,0] # CLS token

                # Roberta Style > add one more dropout layer here
                pooled_output = self.model.dropout(seq_inp)
                pooled_output = self.model.bert.pooler.dense(pooled_output)
                pooled_output = self.model.bert.pooler.activation(pooled_output)
                pooled_output = self.model.dropout(pooled_output)
                logits = self.model.classifier(pooled_output)

        elif self.transformer=='roberta':
            sequence_output = outputs[0]  # (bs, seq_len, dim)
            logits = self.model.classifier(sequence_output)
        else:
            raise Exception("Not Implemented")

        loss = None
        if labels is not None:
            if self.model.config.problem_type is None:
                if self.num_classes > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.model.config.problem_type = "single_label_classification"
                else:
                    self.model.config.problem_type = "multi_label_classification"

            if self.model.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
            elif self.model.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss() # multi label classification, many BCE used
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def get_word_embeddings(self, input_ids=None):
        encoder = getattr(self.model, self.transformer)
        if input_ids is None:
            return encoder.get_input_embeddings()
        else:
            embed_layer = encoder.get_input_embeddings()
            return embed_layer(input_ids)

    def grad_mask(self, input_ids, attention_mask, pred=None, mask_filter=True):
        """
        - Compute masking indices of samples based on gradient signals
        """
        b_length = batch_len(input_ids, self.pad_idx)

        for b_len in b_length:
            if b_len.item() <= 10:
                topk = 1

        delta_grad_ = self.get_emb_grad(input_ids, attention_mask, pred)
        delta_grad = delta_grad_[0].detach()

        # Compute L2-norm of gradients: Saliency Score
        norm_grad = torch.norm(delta_grad, p=2, dim=-1)

        indice_list = []
        for i, len_ in enumerate(b_length):
            if len_>10:
                val, indices_ = torch.topk(norm_grad[i, :len_], 5)
            else:
                val, indices_ = torch.topk(norm_grad[i, :len_], topk)

            last_token = len_.item()-1 # [SEP]
            if mask_filter:
                # if topk indices is not [Cls] and [SEP], then return indice
                ind = [x.item() for x in indices_ if x <= len_.item() and x != last_token and x != 0]
            else:
                ind = [x.item() for x in indices_]

            indice_list.append(ind)

        return indice_list, delta_grad

    def get_emb_grad(self, input_ids, attention_mask, labels=None):
        """
        Get gradient of loss with respect to input tokens embedding.
        Output: [N, seq_length, embedding_size]
        """
        self.model.eval()
        embedding_layer = self.get_word_embeddings()
        original_state = embedding_layer.weight.requires_grad

        embedding_layer.weight.requires_grad = True # allow gradient flow

        # define hook to capture gradient
        emb_grads = []
        def grad_hook(module, grad_in, grad_out):
            emb_grads.append(grad_out[0])

        emb_hook = embedding_layer.register_full_backward_hook(grad_hook)

        self.model.zero_grad()

        output = self.forward(input_ids, attention_mask, labels=labels)

        if labels is None:
            logits = output['logits']
            preds = logits.argmax(dim=-1)
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.num_classes), preds.view(-1))
        else:
            loss = output["loss"]

        loss.backward()
        embedding_layer.weight.requires_grad = original_state
        emb_hook.remove() # Remove Hook
        self.model.eval()

        return emb_grads
    
    def two_step_ensemble(self, input_ids, attention_mask, mask_indices, num_ensemble=5, binom_ensemble=50):
        logit_score_list = []
        
        for _ in range(num_ensemble):
            masked_ids = input_ids.clone()
            for ids_, m_idx in zip(masked_ids, mask_indices):
                for j in range(self.multi_mask):
                    try:
                        ids_[m_idx[j]] = self.mask_idx
                    except:
                        continue
            
            with torch.no_grad():
                output = self.model(masked_ids, attention_mask)
            
            logit_score_list.append(output['logits'])
        
        prediction_score = torch.zeros(input_ids.size(0), self.num_classes).to(self.device)
        
        # create prediction score matrix to store the ensemble output logit
        for logit_ in logit_score_list:
            preds = logit_.argmax(dim=-1)
            prediction_score[range(input_ids.size(0)), preds] += 1
        
        # stack logits from different trial and avg out
        if self.vote_type=='avg':
            logit_tensor_ = torch.stack(logit_score_list)
            logits = logit_tensor_.mean(dim=0)
        else:
            raise Exception("Voting should be Avg")
        
        # n_a store dominant number of predictions
        pred_max = prediction_score.argmax(dim=1)
        n_a = prediction_score[range(input_ids.size(0)), pred_max].long().tolist() 
        for j, n_a_ in enumerate(n_a):
            
            # Test null hypothesis that no less than 50% of samples that are predicted as dominant class
            result = binomtest(n_a_, num_ensemble, p=0.5, alternative='less')
            
            # Hypothesis rejected, Do rand grad mask 
            if result.pvalue < self.alpha_p:
                input_ids_ = input_ids[j].unsqueeze(0)
                attn_m = attention_mask[j].unsqueeze(0)
                
                mask_ind, _ = self.grad_mask(input_ids_, attn_m, pred=None, mask_filter=True)
                self.model.zero_grad()           
                re_ids = input_ids_.repeat(binom_ensemble, 1)
                re_attn = attn_m.repeat(binom_ensemble, 1)
                masked_ids_re = self.masking_single_function(re_ids, mask_ind)
                
                with torch.no_grad():
                    output_re = self.model(masked_ids_re, re_attn)
                
                re_logit = output_re['logits'].mean(dim=0)
                logits[j] = re_logit
        
        return logits

    def masking_single_function(self, input_ids, mask_ind):
        masked_ids = input_ids.clone()
        mask_idx = mask_ind[0]

        if len(mask_ind[0])<self.multi_mask:
            self.multi_mask=1

        for ids_ in masked_ids:
            indices_ = np.random.choice(mask_idx, self.multi_mask)
            for k in indices_:
                try:
                    ids_[k] = self.mask_idx
                except:
                    continue
                    
        return masked_ids


def input_masking_function(input_ids, indices, args):
    """Mask input tokens based on indices computed by grad_mask function"""
    masked_ids = input_ids.clone()

    # mask subset of batch
    if args["mask_batch_ratio"] < 1.0:
        b_size = masked_ids.shape[0]
        b_idx_ = np.arange(b_size)
        np.random.shuffle(b_idx_)
        b_idx = b_idx_[:int(b_size*args.mask_batch_ratio)]

        for idx in b_idx:
            ids_ = masked_ids[idx]
            m_idx = indices[idx]
            for j in range(args["multi_mask"]):
                try:
                    ids_[m_idx[j]] = args["mask_idx"]
                except: # mask id may less than multi_mask number
                    continue
    else:
        for ids_, m_idx in zip(masked_ids, indices): # for each sample in a batch
            for j in range(args["multi_mask"]):
                try:
                    ids_[m_idx[j]] = args["mask_idx"]
                except:
                    continue

    return masked_ids