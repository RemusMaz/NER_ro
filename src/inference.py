
def predict (model, tokenized_input_text):
    # we first have to process our text in the same way we did for training, so let's borrow some code from the Dataset
    input_ids, attention_mask, token_idx = [], [], []

    # let's process each word 
    for i in range(len(tokenized_input_text)):
        token_ids = model.tokenizer.encode(tokenized_input_text[i], add_special_tokens=False)  # tokenize the word, CAREFUL as it could give you 2 or more tokens per word   
        input_ids.extend(token_ids)
        token_idx.extend([i] * len(token_ids))  # save for each added token_id the same word positon i
       
    # the attention mask is now simply a list of 1s the length of the input_ids
    attention_mask = [1] * len(input_ids)

    
    # convert them to tensors; we simulate batches by placing them in [], equivalent to batch_size = 1
    input_ids = torch.tensor([input_ids], device=model.device)  # also place them on the same device (CPU/GPU) as the model
    attention_mask = torch.tensor([attention_mask], device=model.device) 

    # now, we are ready to run the model, but without labels, for which we'll pass None
    with torch.no_grad():
        output = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

    # extract logits and move to cpu
    logits = output['logits'].cpu()  # this will be [1, seq_len, 31], for batch_size = 1, and 31 classes 


    # let's extract our prediction 
    prediction_int = torch.argmax(logits, dim=-1).squeeze().tolist()  # reduce to [seq_len] as list, as batch_size = 1 due to .squeeze()

    word_prediction_int = []
    for i in range(0, max(token_idx) + 1): # for each word in the sentence
        pos = token_idx.index(i)  # find the position of the first ith token, and get pred and gold
        word_prediction_int.append(prediction_int[pos])  # save predicted class

    # last step, convert the ints to strings
    prediction = []
    for i in range(len(word_prediction_int)):
      prediction.append(model.bio2tag_list[word_prediction_int[i]])  # lookup in tag list 
    
    return prediction