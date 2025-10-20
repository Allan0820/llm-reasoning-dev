from transformers import Trainer, TrainingArguments
def tokenize(batch,tokenizer):
    
    inputs = tokenizer(
        batch['natural_language'],
        truncation = True, 
        padding = True #may take slightly more time but ok
    )
    outputs = tokenizer(
        batch['sympy'],
        truncation = True, 
        padding = True
    )
    inputs['labels'] = outputs['input_ids']
    return inputs

def train_model(model, train_tokenized, valid_tokenized, epochs):
    
    train_args = TrainingArguments(
          
        load_best_model_at_end = True, 
        output_dir = "./results",
        per_device_train_batch_size = 8,
        per_device_eval_batch_size = 8,
        gradient_accumulation_steps = 4,
        eval_strategy = 'epoch',
        save_strategy = 'epoch',
        do_eval = True,
        logging_steps = 1,
        eval_steps = 1, 
        num_train_epochs = epochs,
        learning_rate = 2e-5
    )
   
   
    trainer = Trainer(
        
            model = model,
            args = train_args,
            train_dataset = train_tokenized,
            eval_dataset = valid_tokenized,
    
    )
    
    trainer.train()
    

def test_model(model, test_tokenized):
    pass 
