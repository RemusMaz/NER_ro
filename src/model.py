
class TransformerModel(pl.LightningModule):
    def __init__(self, model_name, lr=2e-05, model_max_length=512, bio2tag_list=[], tag_list=[]):
        super().__init__()

        print("Loading AutoModel [{}] ...".format(model_name))
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, strip_accents=False)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(bio2tag_list), from_flax=False)
 
        self.lr = lr
        self.model_max_length = model_max_length
        self.bio2tag_list = bio2tag_list
        self.tag_list = tag_list
        self.num_labels = len(bio2tag_list)

        # we want to record our training loss and validation examples & loss
        # we'll hold them in these lists, and clean them after each epoch
        self.train_loss = []
        self.valid_y_hat = []
        self.valid_y = []
        self.valid_loss = []

    def forward(self, input_ids, attention_mask, labels):
        # we're just wrapping the code on the AutoModelForTokenClassification
        # it needs the input_ids, attention_mask and labels

        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        
        return output["loss"], output["logits"]
        
    def training_step(self, batch, batch_idx):
        # simple enough, just call forward and then save the loss
        loss, _ = self.forward(batch["input_ids"], batch["attention_mask"], batch["labels"])
        self.train_loss.append(loss.detach().cpu().numpy())
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        # call forward to get loss and logits 
        loss, logits = self.forward(batch["input_ids"], batch["attention_mask"], batch["labels"])  # logits is [batch_size, seq_len, num_classes]

        # let's extract our prediction and gold variables - we'll need them to evaluate our predictions
        batch_pred = torch.argmax(logits.detach().cpu(), dim=-1).tolist()  # reduce to [batch_size, seq_len] as list
        batch_gold = batch["labels"].detach().cpu().tolist()  # [batch_size, seq_len] as list
        batch_token_idx = batch["token_idx"].detach().cpu().tolist()

        # because our tokenizer can generate more than one token per word, we'll take the class predicted by the first 
        # token as the class for the word. For example, if we have [Geor] [ge] with predicted classes [B-PERSON] and
        # [B-GPE] (for example), we'll assign to word George the class of [Geor], ignoring any other subsequent tokens.
        batch_size = logits.size()[0]
        for batch_idx in range(batch_size):
            pred, gold, idx = batch_pred[batch_idx], batch_gold[batch_idx], batch_token_idx[batch_idx]
            y_hat, y = [], []
            for i in range(0, max(idx) + 1): # for each sentence, for each word in sequence
                pos = idx.index(i)  # find the position of the first ith token, and get pred and gold
                y_hat.append(pred[pos])  # save predicted class
                y.append(gold[pos])  # save gold class
            self.valid_y_hat.append(y_hat)  
            self.valid_y.append(y)

        self.valid_loss.append(loss.detach().cpu().numpy())  # save our loss as well

        return {"loss": loss}

    def validation_epoch_end(self, outputs):
        print()  # to start printing nicely on a new line
        mean_val_loss = sum(self.valid_loss) / len(self.valid_loss)  # compute average loss
        # for our evaluation, we'll need to convert class indexes to actual classes 
        gold, pred = [], []  
        for y, y_hat in zip(self.valid_y, self.valid_y_hat):  # for each pair of predicted & gold sentences (sequences of ints)
            gold.append([self.bio2tag_list[token_id] for token_id in y])  # go, for each word in the sentence, from class id to class 
            pred.append([self.bio2tag_list[token_id] for token_id in y_hat])  # same for our prediction list

        evaluator = Evaluator(gold, pred, tags=self.tag_list, loader="list")  # call the evaluator 

        # let's print a few metrics
        results, results_by_tag = evaluator.evaluate()
        self.log("valid/avg_loss", mean_val_loss, prog_bar=True)
        self.log("valid/ent_type", results["ent_type"]["f1"])
        self.log("valid/partial", results["partial"]["f1"])
        self.log("valid/strict", results["strict"]["f1"])
        self.log("valid/exact", results["exact"]["f1"])

        # reset our records for a new epoch
        self.valid_y_hat = []
        self.valid_y = []
        self.valid_loss = []

    def configure_optimizers(self):
        return torch.optim.AdamW([p for p in self.parameters() if p.requires_grad], lr=self.lr, eps=1e-08)
