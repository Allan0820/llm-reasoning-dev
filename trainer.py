from transformers import trainer, TrainingArguments
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

def train_model(model, train_tokenized, valid_tokenized):
    
    train_args = TrainingArguments(
          
        load_best_model_at_end = True 
        output_dir = "./results",
        per_device_train_batch_size = 10,
        per_device_eval_batch_size = 10,
        eval_strategy = 'epoch',
        save_strategy = 'epoch',
        do_eval = True,
        logging_steps = 2,
        eval_steps = 2, 
        num_train_epochs = 1,
        learning_rate = 2e-5,
    )
    
    pass 

def test_model(model, test_tokenized):
    pass 
