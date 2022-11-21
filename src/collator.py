import torch


class MyCollator(object):
    def __init__(self, tokenizer, max_seq_len):
        self.max_seq_len = max_seq_len  # this will be our model's maximum sequence length
        self.tokenizer = tokenizer   # we still need our tokenizer to know that the pad token's id is
             

    def __call__(self, input_batch):
        # Question for you: print the input_batch to see what it contains ;)
        output_batch = {
            "input_ids": [],
            "labels": [],
            "token_idx": [],
            "attention_mask": []
        }

        max_len = 0  # we'll need first to find out what is the longest line and then pad the rest to this length 
        
        for instance in input_batch:
            instance_len = min(len(instance["instance_ids"]), self.max_seq_len-2)  # we will never have instances > max_seq_len-2
            max_len = max(max_len, instance_len)  # update max
        
        for instance in input_batch: # for each instance
            instance_ids = instance["instance_ids"]  # it's clearer if we use variables again
            instance_ner_ids = instance[ "instance_ner_ids"]
            instance_token_idx = instance["instance_token_idx"]
            
            # create the attention mask
            # this is a vector of 1s if the token is to be processed (0 if it's padding)
            instance_attention_mask = [1] * len(instance_ids)  # just a list of 1s for now
            
            # cut to max sequence length, if needed
            # notice how easy it is to process them together
            if len(instance_ids) > self.max_seq_len - 2:  # we need the -2 to accomodate for special tokens, this is a transformer's quirk
                instance_ids = instance_ids[:self.max_seq_len - 2]
                instance_ner_ids = instance_ner_ids[:self.max_seq_len - 2]
                instance_token_idx = instance_token_idx[:self.max_seq_len - 2]
                instance_attention_mask = instance_attention_mask[:self.max_seq_len - 2]

            
            """ Depending on your chosen model, the transformer might not have cls and sep, so don't use them 
            if self.tokenizer.cls_token_id and self.tokenizer.sep_token_id:
                instance_ids = [self.tokenizer.cls_token_id] + instance_ids + [self.tokenizer.sep_token_id]
                instance_labels = [0] + instance_labels + [0]
                instance_token_idx = [-1] + instance_token_idx  # no need to pad the last, will do so automatically at return
            """
           
            # how much would we need to pad?
            pad_len = max_len - len(instance_ids)  # with this much

            if pad_len > 0:
                # pad the instance_ids
                instance_ids.extend( [self.tokenizer.pad_token_id] * pad_len)  # notice we're padding with tokenizer.pad_token_id

                # pad the instance_ner_ids
                instance_ner_ids.extend( [0] * pad_len)  # pad with zeros

                # pad the token_ids
                instance_token_idx.extend( [-1] * pad_len)  # notice we're padding with -1 as 0 is a valid word index

                # pad the attention mask
                instance_attention_mask.extend( [0] * pad_len)  # pad with zeros as well

            # add to batch
            output_batch["input_ids"].append(instance_ids)
            output_batch["labels"].append(instance_ner_ids)
            output_batch["token_idx"].append(instance_token_idx)
            output_batch["attention_mask"].append(instance_attention_mask)
      
        # we're done cutting and padding, let's transform them to tensors
        output_batch["input_ids"] = torch.tensor(output_batch["input_ids"])
        output_batch["labels"] = torch.tensor(output_batch["labels"])
        output_batch["token_idx"] = torch.tensor(output_batch["token_idx"])
        output_batch["attention_mask"] = torch.tensor(output_batch["attention_mask"])

        return output_batch
