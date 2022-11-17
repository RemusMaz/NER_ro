class MyDataset(Dataset):
    def __init__(self, tokenizer, instances):
        self.tokenizer = tokenizer  # we'll need this in the __getitem__ function
        self.instances = instances  # save the data for further use

    def __len__(self):
        return len(self.instances)  # return how many instances we have. It's a list after all

    def __getitem__(self, index):
        instance = self.instances[index]  # let's process the ith instance

        # this is what we need to return
        instance_ids, instance_ner_ids, instance_token_idx = [], [], []

        # let's process each word 
        for i in range(len(instance["tokens"])):
            word = instance["tokens"][i]  # this is the ith word in the sentence
            ner_id = instance["ner_ids"][i]  # the ith numeric value corresponding to the ith word

            word_ids = self.tokenizer.encode(word, add_special_tokens=False)  # tokenize the word, CAREFUL as it could give you 2 or more tokens per word
            word_labels = [ner_id] 

            if len(word_ids) > 1:  # we have a word split in more than 1 tokens, fill appropriately
                # the filler will be O, if the class is Other/None, or I-<CLASS>
                if ner_id == 0:  # this is an O, all should be Os
                    word_labels.extend([0] * (len(word_ids) - 1)) 
                else:
                    if word_labels[0] % 2 == 0:  # this is even, so it's an I-<class>, fill with the same Is
                        word_labels.extend([word_labels[0]] * (len(word_ids) - 1))
                    else: # this is a B-<class>, we'll fill it with Is (add 1)
                        word_labels.extend([(word_labels[0]+1)] * (len(word_ids) - 1))

            # add to our instance lists   
            instance_ids.extend(word_ids)  # extend with the token list
            instance_ner_ids.extend(word_labels)  # extend with the ner_id list
            instance_token_idx.extend([i] * len(word_ids))  # extend with the id of the token (to reconstruct words)
        
        return {
            "instance_ids":  instance_ids,
            "instance_ner_ids": instance_ner_ids,
            "instance_token_idx": instance_token_idx
        }