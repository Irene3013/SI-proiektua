import argparse
import sys
import json
import random
import os
from PIL import Image
from collections import Counter

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
# from deepspeed.ops.adam import DeepSpeedCPUAdam

import torch
from torch import optim
from torch.utils import data
from torch.nn import CrossEntropyLoss
import torchvision
from torchvision import transforms, datasets
from transformers.modeling_outputs import SequenceClassifierOutput
import transformers
from transformers import OFATokenizer, OFAModel



## Load model
class LitModel(pl.LightningModule):

    def __init__(self, args):
        super().__init__()
        
        # Load model, tokenizer and loss function
        self.model_name = args.model
        self.model_type = args.target_model
        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.mc_type = None
        if "mc" in self.dataset:
            self.mc_type = int(self.dataset[2])

        if 'OFA' in self.model_name:
            # self.model_name = os.path.join("repos", self.model_name.split("/")[-1])
            self.tokenizer = OFATokenizer.from_pretrained(self.model_name, use_cache=True)
            self.model = OFAModel.from_pretrained(self.model_name, use_cache=True)
            self.loss = CrossEntropyLoss()

        else:
            raise NotImplementedError

        # Define other hyperparameters
        self.warmup_steps = args.warmup_steps
        self.max_steps = args.max_steps
        self.lr = args.lr
        self.opt_eps = args.opt_eps
        self.opt_wd = args.opt_wd
        self.scheduler_off = args.scheduler_off

        # self.deepspeed = args.deepspeed
        self.pretrained_on = None
        self.prev_num_labels = 0

    def configure_optimizers(self):

        # Define optimizer and scheduler
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, eps=self.opt_eps, weight_decay=self.opt_wd)
        if self.scheduler_off:
            return [optimizer]
        else:
            scheduler = {
                "scheduler": transformers.get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=self.max_steps),
                "interval": "step"
            }
            return [optimizer], [scheduler]

    
    def okvqa_accuracy_score(self, y_true, y_pred):
        total_acc = 0
        for out in range(len(y_true)):
            candidates = [y_true[i] for i in range(len(y_true)) if i != out]
            acc = sum(1 for candidate in candidates if candidate == y_pred)
            total_acc += min(acc / 3, 1)
        return total_acc / len(y_true)


    def compute_accuracy_score(self, y_true, y_pred):
        if self.mc_type == 1:
            correct_count = sum([gen.strip().lower() == correct.strip().lower() for gen, correct in zip(y_pred, list(y_true))])
            return correct_count / self.batch_size

        elif self.mc_type == 2:
            0 #TODO
        else:
            total_acc = 0
            for true_targets, pred in zip(y_true, y_pred):
                acc = self.okvqa_accuracy_score(true_targets, pred.lstrip())
                total_acc += acc
            return total_acc / len(y_pred)
    

    def step(self, batch, split):
        
        images, inputs, targets, all_targets = batch

        # Images to device
        patch_images = images.to(self.device)

        # Get input ids
        input_ids = self.tokenizer(inputs, padding=True, truncation=True, return_tensors="pt").input_ids.to(self.device)

        if "mc" in self.dataset:

            # Get decoder input ids ([bos] * batch_size)
            decoder_input_ids = (torch.ones((self.batch_size, 1),  dtype=torch.long) * self.tokenizer.bos_token_id).to(self.device)
            decoder_input_ids = self.tokenizer(targets, padding=True, truncation=True, return_tensors="pt", add_special_tokens=True).input_ids[:, :-1].to(self.device)

            # Get output logits
            outputs = self.model(input_ids, patch_images=patch_images, decoder_input_ids=decoder_input_ids)
            logits = outputs["logits"] 

            #predicted_token_ids = torch.argmax(logits, dim=-1)  # [batch_size, seq_len]

            # Decodificar los tokens a texto
            #predicted_text = self.tokenizer.batch_decode(predicted_token_ids, skip_special_tokens=True)

            # Mostrar las predicciones
            #for i, text in enumerate(predicted_text):
            #    print(f"Sample {i}: {text}")

            #Get label ids
            label_ids = self.tokenizer(targets, padding=True, truncation=True, return_tensors="pt").input_ids[:, 1:].to(self.device)
            """
            # Add padding to logits:
            batch_size, seq_len, vocab_size = logits.shape
            target_seq_len = label_ids.shape[1]
            padding = torch.full((batch_size, target_seq_len - seq_len, vocab_size), fill_value=-1e9, device=self.device)
            logits_padded = torch.cat([logits, padding], dim=1)  

            # Compute loss
            print(f'logits: {logits.shape}, labels: {label.shape}')

            logits = logits_padded.view(-1, logits_padded.size(-1))  # with padding
            """

            #print(f'logits: {logits.shape}, labels: {labels.shape}')

            # Compute loss
            logits = logits.reshape(-1, logits.size(-1)) # without padding       
            labels = label_ids.view(-1)    
            
            loss = self.loss(logits, labels)

            gen_outputs = self.model.generate(input_ids=input_ids, patch_images=patch_images, do_sample=False) #greedy
            y_pred = self.tokenizer.batch_decode(gen_outputs, skip_special_tokens=True)
            y_true = list(targets)
            #print(f'pred: {pred_text} \ncorrect: {correct_choices}')
            #print('-------------------------------------------------\n')

                 
        else:
            
            # Get decoder_input_ids
            decoder_input_ids = self.tokenizer(targets, padding=True, truncation=True, return_tensors="pt", add_special_tokens=True).input_ids[:, :-1].to(self.device)
            
            # Get outputs 
            outputs = self.model(input_ids, patch_images=patch_images, decoder_input_ids=decoder_input_ids)

            # Compute loss
            label_ids = self.tokenizer(targets, padding=True, truncation=True, return_tensors="pt", add_special_tokens=True).input_ids[:, 1:].to(self.device)
            loss = self.loss(outputs["logits"].reshape(-1, outputs["logits"].size(-1)), label_ids.reshape(-1))
              
            # Generate output (to compute accuracy)
            gen_outputs = self.model.generate(input_ids=input_ids, patch_images=patch_images, do_sample=False) #greedy
            y_pred = self.tokenizer.batch_decode(gen_outputs, skip_special_tokens=True)
            y_true = list(zip(*all_targets))
            print(f'logits: {outputs["logits"].shape} \nlabels: {label_ids.shape}\n')

        # Compute Accuracy
        accuracy = self.compute_accuracy_score(y_true, y_pred)
        
        # Save loss
        self.log(f'{split}_loss', loss, on_epoch=True, prog_bar=(split=="train"), logger=True, batch_size=self.batch_size)

        # Save accuracy
        self.log(f'{split}_accuracy', accuracy, on_epoch=True, prog_bar=(split=="train"), logger=True, batch_size=self.batch_size)
  
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, split="train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, split="val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, split="test")
        

## OK-VQA Dataset
class OkVqaDataset(torchvision.datasets.vision.VisionDataset):
    def __init__(self, root: str, split: str, dataset: str, transform = None):
        super().__init__(root, transform=transform)

        # Validations
        assert os.path.exists(root), f"Root directory '{root}' does not exist."
        assert split in ['train', 'val', 'test'], f"Unsupported split: '{split}'. Must be one of ['train', 'val', 'test']."
        assert transform is not None, "Transform cannot be None. Please provide a valid transform."

        self.root = root
        self.split = split
        self.dataset = dataset
        self.transform = transform
        self.all_answers = None
        self.mc_map = {0:"a", 1:"b", 2:"c", 3:"d", 4:"e", 5:"d"}
        self.mc_type = None
        if "mc" in self.dataset:
            self.mc_type = int(self.dataset[2])

        # Load annotations
        self.ann_path = self._get_annotation_path()
        with open(self.ann_path, "r") as f:
            self.annotations = json.load(f)["annotations"]


    def _get_annotation_path(self):
        if "mc" in self.dataset:
            ann_file = f'annotations_{self.split}_mc.json'
        elif self.dataset == "synonyms" and self.split == 'train':
            ann_file = f'annotations_{self.split}_llama3.1:8b.json'
        else:
            ann_file = f'annotations_{self.split}.json'
        return os.path.join(self.root, self.split, ann_file)


    def get_all_answers(self):
        if self.all_answers is None:
            self.all_answers = [str(answer["answer"]) for ann in self.annotations for answer in ann["answers"]]
        return self.all_answers


    def choose_random_answer(self):
        if self.all_answers is None:
            self.get_all_answers()
        return random.choice(self.all_answers)


    def choose_answer(self, answers):
        counts = Counter(answers)
        for threshold in [3, 2]: 
            candidates = [ans for ans, count in counts.items() if count >= threshold]
            if candidates:
                return random.choice(candidates)
        return random.choice(answers)


    def create_mc_input(self, question, choices):
        if self.mc_type:
            str_choices = " \t ".join(choices)
            return f'{question} choose from: {str_choices}\n'
        else:
            str_choices = f"a) {choices[0]} \nb) {choices[1]} \nc) {choices[2]} \nd) {choices[3]} \ne) {choices[4]}"
            return f'{question} choose a, b, c, d or e:\n{str_choices}\n'


    def _load_image(self, image_id):
        image_name = f'okvqa_{self.split}_{str(image_id).zfill(12)}.jpg'
        img_path = os.path.join(self.root, self.split, self.split, image_name)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = Image.open(img_path).convert("RGB")
        return self.transform(image)


    def __getitem__(self, index):
        annotation = self.annotations[index]
        image = self._load_image(annotation["image_id"])
        question = annotation["question"]

        if "mc" in self.dataset:
            answer_choices = annotation["choices"]
            if self.mc_type: 
                correct_choice = annotation["correct_choice"]
            else: 
                correct_choice = self.mc_map[annotation["correct_choice_idx"]]
            return image, self.create_mc_input(question, answer_choices), correct_choice, 0

        answers = [str(ans["answer"]) for ans in annotation["answers"]]
        if self.split == 'train' and self.dataset == "random":
            random_answers = [self.choose_random_answer()]
            return image, question, self.choose_answer(random_answers), random_answers

        return image, question, self.choose_answer(answers),  answers


    def __len__(self):
        return len(self.annotations)

## Data Module
class OKVQADataModule(pl.LightningDataModule):

    def __init__(self, args):
        super().__init__()

        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.root = args.root
        self.visual_root = args.visual_root
        self.variant = args.vsr_variant
        self.target_model = args.target_model
        self.grid_size = args.grid_size
        self.locations = args.location_encoding
        self.dataset = args.dataset
        self.attributes = args.attributes
        self.model_name = args.model

        if 'huge' in self.model_name or 'large' in self.model_name:
            resolution = 480
        elif 'base' in self.model_name:
            resolution = 384
        elif 'tiny' in self.model_name or 'medium' in self.model_name:
            resolution = 256
        else:
            raise NotImplementedError

        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def train_dataloader(self):
        split = 'train'
        dataset = OkVqaDataset(root=self.root, split=split, dataset=self.dataset, transform=self.transform)
        params = {
            'batch_size': self.batch_size,
            'shuffle': True,
            'num_workers': self.num_workers,
        }
        return data.DataLoader(dataset, **params)

    def val_dataloader(self):
        split = 'val'
        dataset = OkVqaDataset(root=self.root, split=split, dataset=self.dataset, transform=self.transform)
        params = {
            'batch_size': self.batch_size,
            'shuffle': False,
            'num_workers': self.num_workers,
        }
        return data.DataLoader(dataset, **params)

    def test_dataloader(self):
        split = 'test'
        dataset = OkVqaDataset(root=self.root, split=split, dataset=self.dataset, transform=self.transform)
        params = {
            'batch_size': self.batch_size,
            'shuffle': False,
            'num_workers': self.num_workers,
        }
        return data.DataLoader(dataset, **params)

## Parse arguments
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model", type=str, required=True, choices=["unc-nlp/lxmert-base-uncased", "unc-nlp/lxmert-vqa-uncased", "dandelin/vilt-b32-finetuned-vqa", "dandelin/vilt-b32-mlm", "OFA-Sys/ofa-tiny", "OFA-Sys/ofa-medium", "OFA-Sys/ofa-base", "OFA-Sys/ofa-large", "OFA-Sys/ofa-huge"],
        help="Model type to be fine-tuned."
    )
    parser.add_argument(
        "--ckpt", type=str, default=None, help="Model's checkpoint to be loaded before training."
    )
    parser.add_argument(
        "--gpus", type=int, default=1, help="Number of GPUs in use. (0 == cpu)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=56, help="Batch size (per gpu)."
    )
    parser.add_argument(
        "--accumulate_grad_batches", type=int, default=1, help="Gradient accumulation steps. (1 == do not use gradient accumulation)"
    )
    parser.add_argument(
        "--scheduler_off", action="store_true", help="Do not use any scheduler"
    )
    parser.add_argument(
        "--deepspeed", action="store_true", help="Use deepspeed stage-2 offload strategy."
    )
    parser.add_argument(
        "--val_check_interval", type=float, default=1.0, help="How often within a training epoch to check the val set. (1.0 == every epoch)"
    )
    parser.add_argument(
        "--lr", type=float, default=5e-5, help="Learning rate."
    )
    parser.add_argument(
        "--precision", type=int, default=32, choices=[16, 32, 64], help="Precision for the GPUs."
    )
    parser.add_argument(
        "--opt_eps", type=float, default=1e-8, help="Epsilon value for AdamW optimizer."
    )
    parser.add_argument(
        "--opt_wd", type=float, default=0.0, help="Weight decay value for AdamW optimizer."
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=2000, help="Warmup steps to be done during training."
    )
    parser.add_argument(
        "--max_steps", type=int, default=88000, help="Steps to be done during training."
    )
    parser.add_argument(
        "--dataset", type=str, default="original", choices=["original", "random", "synonyms", "mc1", "mc2"],
        help="Select dataset to be trained on."
    )
    parser.add_argument(
        "--root", type=str, default="/gscratch3/users/gazkune/datasets/vsr/vsr_seq2seq_files", help="Path to the Coco or VinVL prediction files."
    )
    # For coco: "/ikerlariak/asalaberria009/datasets/mscoco"
    parser.add_argument(
        "--visual_root", type=str, default="/ikerlariak/asalaberria009/datasets/mscoco/images", help="Path to the Coco or VinVL prediction files."
    )
    parser.add_argument(
        "--vsr_variant", type=str, default="random", choices=["random", "zero-shot"], help="Variant of the VSR dataset."
    )
    parser.add_argument(
        "--source", type=str, default="vinvl", choices=["coco", "vinvl"], help="Source of the object annotations."
    )
    parser.add_argument(
        "--target_model", type=str, default="bert", choices= ["bert", "t5", "lxmert", "vilt", "ofa"], help="Generate inputs and outputs for a specific LM."
    )
    parser.add_argument(
        "--location_encoding", type=str, default="none", choices= ["none", "token", "grid", "rect", "none"], help="What kind of spatial representation to use."
    )
    parser.add_argument(
        "--distractors", type=int, default=-1, help="How many objects we should use as distractors (-1: all available)."
    )
    parser.add_argument(
        "--attributes", action="store_true", help="Use VinVL attributes for image descriptions."
    )
    parser.add_argument(
        "--spatial_val_file", type=str, default="/gscratch3/users/gazkune/datasets/vinvl_vqa/validation-vinvl-alldistractors-noattr.json", help="Use an already prepared spatial validation file; if None, it will be generated on the fly."
    )
    # Use /gscratch3/users/gazkune/datasets/vinvl_vqa/validation-vinvl-alldistractors-nolocation.json to ignore locations
    # Use /gscratch3/users/gazkune/datasets/vinvl_vqa/validation-vinvl-alldistractors-noattr-nolocation.json to ignore attributes and locations
    parser.add_argument(
        "--tiny", action="store_true", help="Use tiny version of the dataset for development."
    )
    parser.add_argument(
        "--num_workers", type=int, default=12, help="Workers used in the dataloader." 
    )

    parser.add_argument(
        "--grid_size", type=int, default=32, help="The size of the grid for the location encoding."
    )

    parser.add_argument(
        "--seed", type=int, default=-1, help="Seed."
    )

    parser.add_argument(
        "--train", action="store_true", help="Fine-tune model."
    )
    parser.add_argument(
        "--evaluate", action="store_true", help="Test model after fine-tuning."
    )
    parser.add_argument(
        "--run_name", type=str, default=None, help="Name of the run. Used in tensorboard and output filenames. If it is not filled or already exists, a custom one will be generated."
    )
    parser.add_argument(
        "--output_path", type=str, default="/gaueko0/users/ietxarri010/ofa_okvqa_finetuning/", help="Output directory for plots and models."
    )

    args = parser.parse_args()
    return args
        
## Main program
def main():

    print("Parsing args...")
    args = parse_args()
    # Reproducibility
    if args.seed != -1:
        pl.utilities.seed.seed_everything(args.seed)

    # Load model
    print("Loading model...")
    if args.ckpt is None:
        model = LitModel(args)
    else:
        model = LitModel.load_from_checkpoint(checkpoint_path=args.ckpt, args=args, strict=False)

    print("Model loaded!")

    # Load data
    print("Loading dataset...")

    #if args.dataset == 'spatialcoco': #DEFAULT: spatialcoco
    datamodule = OKVQADataModule(args)
       
    print("Dataset loaded!")

    # Define checkpoint filename and tensorboard run name
    if args.run_name == None:
        print('A run name has to be provided')
        sys.exit()

    tb_run_name = args.run_name
    print(f'Run name: {tb_run_name}')
    
    # Define deepspeed strategy
    #deepspeed = None
    deepspeed = 'auto'
    # if args.deepspeed and args.gpus > 1:
    #     deepspeed="deepspeed_stage_2_offload"
    

    # Define trainer
    print(f'gpus: {args.gpus}')
    
    logger = TensorBoardLogger("logs", name=tb_run_name, default_hp_metric=False)
    
    # Use ModelCheckPoint to store best validation model
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_path, 
        monitor='val_accuracy', 
        mode='max', 
        filename=tb_run_name + "-{epoch:02d}-{val_accuracy:.2f}", 
        save_weights_only=True, 
        save_top_k=1)
        
    if args.model == "t5-11b":
        trainer = pl.Trainer(
            devices=args.gpus, 
            fast_dev_run=False, 
            logger=logger, 
            max_steps=args.max_steps, 
            accumulate_grad_batches=args.accumulate_grad_batches, 
            strategy=deepspeed, 
            precision=args.precision, 
            callbacks=[checkpoint_callback])
    
    else:
        trainer = pl.Trainer(
            callbacks=[checkpoint_callback],
            devices=args.gpus, 
            fast_dev_run=False, 
            logger=logger, 
            max_steps=args.max_steps,
            accumulate_grad_batches=args.accumulate_grad_batches, 
            strategy=deepspeed, 
            precision=args.precision)
    # NOTE: accumulate_grad_batches=4 . Ex: to have a batch size of 56, I have to use 14 (56/4)
    # NOTE: val_check_interval -> if float (percentage of epoch); if int, number of steps to run validation

    # Train model
    if args.train:
        print("Training starts!")
        trainer.fit(model, datamodule)
        print("Training finished!")

    # Evaluate model
    if args.evaluate and args.train:
        print(f'Loading {checkpoint_callback.best_model_path} with val accuracy of {checkpoint_callback.best_model_score} to test')
        print('Testing starts!')
        trainer.test(ckpt_path = 'best', dataloaders=datamodule.test_dataloader(), verbose=False)
        print('Testing finished!')
    elif args.evaluate and not args.train:
        print('Testing starts!')
        trainer.test(model=model, dataloaders=datamodule.test_dataloader(), verbose=False)
        print('Testing finished!')
    
    return 0

if __name__ == "__main__":
    main()
