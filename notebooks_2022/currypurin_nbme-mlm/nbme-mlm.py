#!/usr/bin/env python
# coding: utf-8

# ref: https://www.kaggle.com/competitions/nbme-score-clinical-patient-notes/discussion/323095

# In[ ]:


# The following is necessary if you want to use the fast tokenizer for deberta v2 or v3
# This must be done before importing transformers
import shutil
from pathlib import Path

transformers_path = Path("/opt/conda/lib/python3.7/site-packages/transformers")

input_dir = Path("../input/deberta-v2-3-fast-tokenizer")

convert_file = input_dir / "convert_slow_tokenizer.py"
conversion_path = transformers_path/convert_file.name

if conversion_path.exists():
    conversion_path.unlink()

shutil.copy(convert_file, transformers_path)
deberta_v2_path = transformers_path / "models" / "deberta_v2"

for filename in ['tokenization_deberta_v2.py', 'tokenization_deberta_v2_fast.py']:
    filepath = deberta_v2_path/filename
    
    if filepath.exists():
        filepath.unlink()

    shutil.copy(input_dir/filename, filepath)


# In[ ]:


get_ipython().run_cell_magic('writefile', 'mlm.py', '\nimport argparse\nimport os\nimport json\nfrom pathlib import Path\n\nimport pandas as pd\nfrom tqdm.auto import tqdm\nimport torch\nfrom datasets import load_dataset\nimport tokenizers\nimport transformers\nfrom transformers import AutoTokenizer, AutoConfig\nfrom transformers import DataCollatorForLanguageModeling, AutoModelForMaskedLM, Trainer\nfrom transformers import TrainingArguments\nfrom transformers.utils import logging\nfrom IPython import embed  # noqa\n\nlogging.set_verbosity_info()\nlogger = logging.get_logger(__name__)\nlogger.info("INFO")\nlogger.warning("WARN")\nKAGGLE_ENV = True if \'KAGGLE_URL_BASE\' in set(os.environ.keys()) else False\n\n\nprint(f"tokenizers.__version__: {tokenizers.__version__}")\nprint(f"transformers.__version__: {transformers.__version__}")\ndevice = torch.device(\'cuda\' if torch.cuda.is_available() else \'cpu\')\nINPUT_DIR = Path(\'../input/\')\nif KAGGLE_ENV:\n    OUTPUT_DIR = Path(\'\')\n    os.environ["WANDB_DISABLED"] = "true"\nelse:\n    OUTPUT_DIR = INPUT_DIR\n\n\ndef get_patient_notes_not_used_train():\n\n    patient_notes = pd.read_csv(INPUT_DIR / \'nbme-score-clinical-patient-notes\' / "patient_notes.csv")\n    print(patient_notes.shape)\n    train = pd.read_csv(INPUT_DIR / \'nbme-score-clinical-patient-notes\' / \'train.csv\')\n    train_pn_num_unique = train[\'pn_num\'].unique()\n\n    train_patient_notes = \\\n        patient_notes.loc[~patient_notes[\'pn_num\'].isin(train_pn_num_unique), :].reset_index(drop=True)\n    valid_patient_notes = \\\n        patient_notes.loc[patient_notes[\'pn_num\'].isin(train_pn_num_unique), :].reset_index(drop=True)\n\n    print(train_patient_notes.shape)\n    print(valid_patient_notes.shape)\n    return train_patient_notes, valid_patient_notes\n\n\ndef tokenize_function(examples):\n    return tokenizer(examples["text"])\n\n\ndef get_tokenizer(args):\n    if \'v3\' in str(args.model_path):\n        from transformers.models.deberta_v2.tokenization_deberta_v2_fast import DebertaV2TokenizerFast\n        print(\'DebertaV2TokenizerFast\')\n        tokenizer = DebertaV2TokenizerFast.from_pretrained(INPUT_DIR / args.model_path, trim_offsets=False)\n    else:\n        if args.model_name:\n            print(\'model_name\', args.model_name)\n            tokenizer = AutoTokenizer.from_pretrained(args.model_name, trim_offsets=False)\n        else:\n            print(\'model_path\', args.model_path)\n            tokenizer = AutoTokenizer.from_pretrained(INPUT_DIR / args.model_path, trim_offsets=False)\n    return tokenizer\n\n\ndef parse_args():\n    parser = argparse.ArgumentParser()\n    parser.add_argument("--model_name", type=str, default="", required=False)\n    parser.add_argument("--model_path", type=str, default="../input/deberta-v3-large/deberta-v3-large/", required=False)\n    parser.add_argument("--seed", type=int, default=0, required=False)\n    parser.add_argument(\'--debug\', action=\'store_true\', required=False)\n    parser.add_argument(\'--exp_num\', type=str, required=True)\n    parser.add_argument("--param_freeze", action=\'store_true\', required=False)\n    parser.add_argument("--num_train_epochs", type=int, default=5, required=False)\n    parser.add_argument("--batch_size", type=int, default=8, required=False)\n    parser.add_argument("--lr", type=float, default=2e-5, required=False)\n    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, required=False)\n    return parser.parse_args()\n\n\nif __name__ == "__main__":\n\n    args = parse_args()\n    train, valid = get_patient_notes_not_used_train()\n\n    if args.debug:\n        train = train.iloc[:10, :]\n        valid = valid.iloc[:10, :]\n        args.batch_seize = 1\n\n    def get_text(df):\n        text_list = []\n        for text in tqdm(df[\'pn_history\']):\n            if len(text) < 30:\n                pass\n            else:\n                text_list.append(text)\n        return text_list\n\n    train_text_list = get_text(train)\n    valid_text_list = get_text(valid)\n\n    mlm_train_json_path = OUTPUT_DIR / \'train_mlm.json\'\n    mlm_valid_json_path = OUTPUT_DIR / \'valid_mlm.json\'\n\n    for json_path, list_ in zip([mlm_train_json_path, mlm_valid_json_path],\n                                [train_text_list, valid_text_list]):\n        with open(str(json_path), \'w\') as f:\n            for sentence in list_:\n                row_json = {\'text\': sentence}\n                json.dump(row_json, f)\n                f.write(\'\\n\')\n\n    datasets = load_dataset(\n        \'json\',\n        data_files={\'train\': str(mlm_train_json_path),\n                    \'valid\': str(mlm_valid_json_path)},\n        )\n\n    if mlm_train_json_path.is_file():\n        mlm_train_json_path.unlink()\n    if mlm_valid_json_path.is_file():\n        mlm_valid_json_path.unlink()\n    print(datasets["train"][:2])\n\n    tokenizer = get_tokenizer(args)\n\n    tokenized_datasets = datasets.map(\n        tokenize_function,\n        batched=True,\n        num_proc=1,\n        remove_columns=["text"],\n        batch_size=args.batch_size)\n    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)\n\n    if args.model_name:\n        print(\'model_name:\', args.model_name)\n        model_name = args.model_name\n    else:\n        print(\'model_path:\', args.model_path)\n        model_name = INPUT_DIR / args.model_path\n    config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)\n\n    if \'v3\' in str(model_name):\n        model = transformers.DebertaV2ForMaskedLM.from_pretrained(INPUT_DIR / model_name, config=config)\n    else:\n        model = AutoModelForMaskedLM.from_pretrained(model_name, config=config)\n\n    if args.param_freeze:\n        # if freeze, Write freeze settings here\n\n        # deberta-v3-large\n        # model.deberta.embeddings.requires_grad_(False)\n        # model.deberta.encoder.layer[:12].requires_grad_(False)\n\n        # deberta-large\n        model.deberta.embeddings.requires_grad_(False)\n        model.deberta.encoder.layer[:24].requires_grad_(False)\n\n        for name, p in model.named_parameters():\n            print(name, p.requires_grad)\n\n    if args.debug:\n        save_steps = 100\n        args.num_train_epochs = 1\n    else:\n        save_steps = 100000000\n\n    training_args = TrainingArguments(\n        output_dir="output-mlm",\n        evaluation_strategy="epoch",\n        learning_rate=args.lr,\n        weight_decay=0.01,\n        save_strategy=\'no\',\n        per_device_train_batch_size=args.batch_size,\n        num_train_epochs=args.num_train_epochs,\n        # report_to="wandb",\n        run_name=f\'output-mlm-{args.exp_num}\',\n        # logging_dir=\'./logs\',\n        lr_scheduler_type=\'cosine\',\n        warmup_ratio=0.2,\n        fp16=True,\n        logging_steps=500,\n        gradient_accumulation_steps=args.gradient_accumulation_steps\n    )\n\n    trainer = Trainer(\n        model=model,\n        args=training_args,\n        train_dataset=tokenized_datasets["train"],\n        eval_dataset=tokenized_datasets[\'valid\'],\n        data_collator=data_collator,\n        # optimizers=(optimizer, scheduler)\n    )\n\n    trainer.train()\n\n    if args.model_name == \'microsoft/deberta-xlarge\':\n        model_name = \'deberta-xlarge\'\n    elif args.model_name == \'microsoft/deberta-large\':\n        model_name = \'deberta-large\'\n    elif args.model_name == \'microsoft/deberta-base\':\n        model_name = \'deberta-base\'\n    elif args.model_path == "../input/deberta-v3-large/deberta-v3-large/":\n        model_name = \'deberta-v3-large\'\n    elif args.model_name == \'microsoft/deberta-v2-xlarge\':\n        model_name = \'deberta-v2-xlarge\'\n    trainer.model.save_pretrained(OUTPUT_DIR / f\'{args.exp_num}_mlm_{model_name}\')\n')


# In[ ]:


get_ipython().system('python mlm.py --debug --exp_num 0')


# In[ ]:


ls 


# In[ ]:




