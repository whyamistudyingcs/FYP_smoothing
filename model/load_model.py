from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

def load_base_model(args):
    print(f"Load model...")
    if  args["model"] == 'bert':
        from transformers import BertForSequenceClassification
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=args["num_classes"])
    elif  args["model"] == 'bert-large':
        from transformers import BertForSequenceClassification
        model = BertForSequenceClassification.from_pretrained("bert-large-uncased", num_labels=args["num_classes"])
    elif  args["model"] == 'roberta':
        from transformers import RobertaForSequenceClassification
        model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=args["num_classes"])
    elif  args["model"] == 'roberta-large':
        from transformers import RobertaForSequenceClassification
        model = RobertaForSequenceClassification.from_pretrained('roberta-large', num_labels=args["num_classes"])
    else:
        raise Exception("Specify model correctly...")
    model.config.num_labels = args["num_classes"]
    return model


def load_tokenizer(args):
    print(f"Load Tokenizer...")
    if  args["model"] == 'bert':
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    elif  args["model"] == 'bert-large':
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    elif  args["model"] == 'roberta':
        from transformers import RobertaTokenizer
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    elif  args["model"] == 'roberta-large':
        from transformers import RobertaTokenizer
        tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    else:
        raise Exception("Specify model correctly...")

    pad_idx = tokenizer.pad_token_id
    mask_idx = tokenizer.mask_token_id
    print(f"Tokenizer: { args['model']} || PAD: {pad_idx} || MASK: {mask_idx}")
    return tokenizer


def noisy_forward_loader(args):
    if 'roberta' in  args["model"]:
        from transformers import RobertaForSequenceClassification

        if  args["model"] == 'roberta-large':
            model = RobertaForSequenceClassification.from_pretrained('roberta-large', num_labels=args["num_classes"])
        else:
            model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=args["num_classes"])

        import types
        model.roberta.encoder.forward = types.MethodType(roberta_noise_forward, model.roberta.encoder)

    elif 'bert' in  args["model"]:
        from transformers import BertForSequenceClassification

        if  args["model"] == 'bert-large':
            model = BertForSequenceClassification.from_pretrained('bert-large-uncased', num_labels=args["num_classes"])
        else:
            model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=args["num_classes"])

        import types
        model.bert.encoder.forward = types.MethodType(bert_noise_forward, model.bert.encoder)

    else:
        raise Exception("Specify Base model correctly...")

    model.config.single_layer = args["single_layer"]
    model.config.num_labels = args["num_classes"]
    model.config.nth_layers = args["nth_layers"]
    model.config.noise_eps = args["noise_eps"]

    return model

def bert_noise_forward(
    self,
    hidden_states,
    attention_mask=None,
    head_mask=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
    past_key_values=None,
    use_cache=None,
    output_attentions=False,
    output_hidden_states=False,
    return_dict=True,
):
    all_hidden_states = () if output_hidden_states else None
    all_self_attentions = () if output_attentions else None
    all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

    next_decoder_cache = () if use_cache else None

    for i, layer_module in enumerate(self.layer):
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        layer_head_mask = head_mask[i] if head_mask is not None else None
        past_key_value = past_key_values[i] if past_key_values is not None else None

        layer_outputs_ = layer_module(
            hidden_states,
            attention_mask,
            layer_head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )

        if i%self.config.nth_layers==0 and self.config.single_layer==False:
            randn_noise = torch.randn_like(layer_outputs_[0])*self.config.noise_eps
            temp  = layer_outputs_[0]+randn_noise

            if output_attentions:
                layer_outputs = (temp, layer_outputs_[1])
            else:
                layer_outputs = tuple(temp[None,:])

        else:
            layer_outputs = layer_outputs_

        hidden_states = layer_outputs[0]
        if use_cache:
            next_decoder_cache += (layer_outputs[-1],)
        if output_attentions:
            all_self_attentions = all_self_attentions + (layer_outputs[1],)
            if self.config.add_cross_attention:
                all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    if not return_dict:
        return tuple(
            v for v in [
                hidden_states,
                next_decoder_cache,
                all_hidden_states,
                all_self_attentions,
                all_cross_attentions,
            ]
            if v is not None
        )

    return BaseModelOutputWithPastAndCrossAttentions(
        last_hidden_state=hidden_states,
        past_key_values=next_decoder_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
        cross_attentions=all_cross_attentions,
    )

def roberta_noise_forward(
    self,
    hidden_states,
    attention_mask=None,
    head_mask=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
    past_key_values=None,
    use_cache=None,
    output_attentions=False,
    output_hidden_states=False,
    return_dict=True,
):
    all_hidden_states = () if output_hidden_states else None
    all_self_attentions = () if output_attentions else None
    all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

    next_decoder_cache = () if use_cache else None
    
    for i, layer_module in enumerate(self.layer):
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        layer_head_mask = head_mask[i] if head_mask is not None else None
        past_key_value = past_key_values[i] if past_key_values is not None else None
        
        # Forward pass through the layer
        layer_outputs_ = layer_module(
                hidden_states, # torch.Size([16, 80, 768])
                attention_mask, # torch.Size([16, 1, 1, 80])
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
        )

        # Inject noise if conditions are met
        if i%self.config.nth_layers==0 and self.config.single_layer==False:
            #print(f"Multi Layer: {i}-layer")
            randn_noise = torch.randn_like(layer_outputs_[0])*self.config.noise_eps

            temp  = layer_outputs_[0]+randn_noise

            if output_attentions:
                layer_outputs = (temp, layer_outputs_[1])
            else:
                layer_outputs = tuple(temp[None,:])

        else:
            layer_outputs = layer_outputs_

        hidden_states = layer_outputs[0]
        
        if use_cache:
            next_decoder_cache += (layer_outputs[-1],)
        if output_attentions:
            all_self_attentions = all_self_attentions + (layer_outputs[1],)
            if self.config.add_cross_attention:
                all_cross_attentions = all_cross_attentions + (layer_outputs[2],)
    
    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)
    

    if not return_dict:
        return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)

    return BaseModelOutputWithPastAndCrossAttentions(
        last_hidden_state=hidden_states,
        past_key_values=next_decoder_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
        cross_attentions=all_cross_attentions,
    )
