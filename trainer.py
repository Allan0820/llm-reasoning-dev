from transformers import Trainer, TrainingArguments

def tokenize(batch,tokenizer):
    
    inputs = tokenizer(
        batch['natural_language'],
        truncation = True, 
        max_length = 128,
        padding = 'max_length'
    
    )
    outputs = tokenizer(
        batch['sympy'],
        truncation = True, 
        max_length = 128,
        padding = 'max_length'

    )
    inputs['labels'] = outputs['input_ids']
    return inputs

def train_model(model, train_tokenized, valid_tokenized, epochs):
    
    train_args = TrainingArguments(
          
        output_dir = f"./results/llama7b",
        per_device_train_batch_size = 10,
        per_device_eval_batch_size = 10,
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
