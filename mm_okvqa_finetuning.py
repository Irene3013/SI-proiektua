import argparse
import sys
import json
import random
import os
from PIL import Image

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger
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

def to_one_hot(y_tensor, n_dims=None):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    return y_one_hot

def okvqa_accuracy_score(y_true, y_pred):
    total_acc = 0
    for out in range(len(y_true)):
        candidates = [y_true[i] for i in range(len(y_true)) if i != out]
        acc = sum(1 for candidate in candidates if candidate == y_pred)
        total_acc += min(acc / 3, 1)
    return total_acc / len(y_true)

def preprocess_prediction(pred):
    return pred.lstrip()

def compute_accuracy_score(y_true, y_pred):
    total_acc = 0
    register = []
    for true_targets, pred in zip(y_true, y_pred):
        acc = okvqa_accuracy_score(true_targets, preprocess_prediction(pred))
        register.append([true_targets, pred, acc])
        total_acc += acc
    return total_acc / len(y_pred), register

  
## Load model
class LitModel(pl.LightningModule):

    def __init__(self, args):
        super().__init__()

        # MOVE: Load task labels
        self.dataset = args.dataset # NOTE: 'spatialcoco'

        if self.dataset == 'spatialcoco':
            self.labels = ['yes', 'no']
            self.num_labels = 1 # NOTE: it's binary classification
            print(f'Number of labels: {self.num_labels}')

        # Load model, tokenizer and loss function
        self.model_name = args.model
        self.model_type = args.target_model

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
        
        """"
        #RandomSearch: Model.generate

        self.iteration = args.iteration
        self.search_space = {
            'num_beams': [5, 7, 10],
            'no_repeat_ngram_size': [1],
            'repetition_penalty': [1.5, 2.0],
            'temperature': [0.7, 0.8, 1.0],
            'top_k': [10, 20, 50],
            'top_p': [0.7, 0.8],
            'max_length': [20, 50]
        }

        self.params = {key: random.choice(values) for key, values in self.search_space.items()}
        """
        self.params = {
            'num_beams': 5,
            'no_repeat_ngram_size': 1,
            'repetition_penalty': 1.5,
            'temperature': 1.0,
            'top_k': 50,
            'top_p': 0.7,
            'max_length': 20
        }

        # output
        self.predictions = f"./output/Predictions/predictions_{self.iteration}.txt"
        self.info = f"./output/Info/info.txt_{self.iteration}.txt"
        self.rs = f"./output/RandomSearch/RandomSearch_{self.iteration}.txt"

        # create directory if needed
        os.makedirs(os.path.dirname(self.predictions), exist_ok=True)
        os.makedirs(os.path.dirname(self.info), exist_ok=True)
        os.makedirs(os.path.dirname(self.rs), exist_ok=True)

        # delete file content or create new file
        with open(self.predictions, 'w') as f: pass
        with open(self.info, 'w') as f: pass
        with open(self.rs, 'w') as f: pass

        with open(self.rs, 'a') as f0:
          f0.write(f'ITERATION {self.iteration} \n')
          f0.write(f'\nPARAMS:\n')
          f0.write(f'-> num_beams:  {self.params["num_beams"]} \n')
          f0.write(f'-> no_repeat_ngram_size:  {self.params["no_repeat_ngram_size"]} \n')
          f0.write(f'-> repetition_penalty:  {self.params["repetition_penalty"]} \n')
          f0.write(f'-> temperature:  {self.params["temperature"]} \n')
          f0.write(f'-> top_k:  {self.params["top_k"]} \n')
          f0.write(f'-> top_p:  {self.params["top_p"]} \n')
          f0.write(f'-> max_length:  {self.params["max_length"]} \n \n')

          f0.write(f'\nWEIGHT DECAY: {self.opt_wd}\n')
        
    def configure_optimizers(self):
        # Define optimizer and scheduler
        """
        if self.deepspeed:
            optimizer = DeepSpeedCPUAdam(self.model.parameters(), lr=self.lr, eps=self.opt_eps, weight_decay=self.opt_wd)
        else:
            optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, eps=self.opt_eps, weight_decay=self.opt_wd)
        if self.scheduler_off:
            return [optimizer]
        else:
            scheduler = {
                "scheduler": transformers.get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=self.max_steps),
                "interval": "step"
            }
            return [optimizer], [scheduler]
        """
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, eps=self.opt_eps, weight_decay=self.opt_wd)
        if self.scheduler_off:
            return [optimizer]
        else:
            scheduler = {
                "scheduler": transformers.get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=self.max_steps),
                "interval": "step"
            }
            return [optimizer], [scheduler]


    def step(self, batch, split):

        images, questions, all_targets, targets = batch
        
        if self.model_type == "ofa":
            
            # Images to device
            patch_images = images.to(self.device)
            
            # Get input ids
            input_ids = self.tokenizer(questions, padding=True, truncation=True, return_tensors="pt").input_ids.to(self.device)
            
            # Get decoder_input_ids
            decoder_input_ids = self.tokenizer(targets, padding=True, truncation=True, return_tensors="pt", add_special_tokens=True).input_ids[:, 1:].to(self.device)
            
            # Compute loss
            outputs = self.model(input_ids, patch_images=patch_images, decoder_input_ids=decoder_input_ids)
            label_ids = self.tokenizer(targets, padding=True, truncation=True, return_tensors="pt", add_special_tokens=True).input_ids.to(self.device)

            # Register info
            with open(self.info, 'a') as f1:

              batch_size, seq_len, _ = outputs["logits"].shape
              _, label_seq_len = label_ids[:, 1:].shape

              for example_idx in range(batch_size):  # images.size(0) es el tamaño del lote

                  # Iterar a través de todas las posiciones de la secuencia
                  for sequence_idx in range(seq_len):  # input_ids.size(1) es la longitud de la secuencia
                      if sequence_idx >= label_seq_len:  # Verificar que no estamos fuera del rango de `label_ids`
                          continue
                      
                      # Obtener los logits y etiquetas antes del reshape
                      logits_example = outputs["logits"][example_idx, sequence_idx]
                      label_example = label_ids[example_idx, sequence_idx+1]

                      max_len = 10 + len("Pred: ") + 2

                      # Decodificar para verificar
                      pred_token = torch.argmax(logits_example).item()
                      pred_word = self.tokenizer.decode([pred_token])
                      true_word = self.tokenizer.decode([label_example.item()])

                      pred_str = f"Pred: {pred_word}".ljust(max_len + 2)
                      true_str = f"True: {true_word}"

                      # Escribir en el archivo
                      f1.write(f"{pred_str}{true_str}\n")
                  f1.write(f"\n")

              loss = self.loss(outputs["logits"].reshape(-1, outputs["logits"].size(-1)), label_ids[:, 1:].reshape(-1))

              f1.write(f"LOSS: {loss} \n")
              f1.write(f'------------------------- \n')

            # Generate output (to compute accuracy)
            gen_outputs = self.model.generate(
                    input_ids=input_ids,
                    patch_images=patch_images,
                    num_beams=self.params['num_beams'],
                    no_repeat_ngram_size=self.params['no_repeat_ngram_size'],
                    repetition_penalty=self.params['repetition_penalty'],
                    temperature=self.params['temperature'],
                    top_k=self.params['top_k'],
                    top_p=self.params['top_p'],
                    max_length=self.params['max_length'],
                    early_stopping=True
            )
            pred_text = self.tokenizer.batch_decode(gen_outputs, skip_special_tokens=True)
        
        else:

            raise NotImplementedError

        # Save Loss
        self.log(f'{split}_loss', loss, on_epoch=True, prog_bar=(split=="train"), logger=True, batch_size=len(questions))
        
        # Compute Accuracy
        accuracy, register = compute_accuracy_score(list(zip(*all_targets)), pred_text)
        self.log(f'{split}_accuracy', accuracy, on_epoch=True, prog_bar=(split=="train"), logger=True, batch_size=len(questions))

        # Register predictons
        with open(self.predictions, 'a') as f2:
          for reg in register:
            f2.write(f'\n True labels: {reg[0]} \n Prediction: {reg[1]} \n Accuracy: {reg[2]} \n')
          f2.write(f'BATCH ACCURACY {accuracy} \n')
          f2.write(f'------------------------- \n')
        
        with open(self.rs, 'a') as f0:
          f0.write(f'LOSS {loss} \t ACC {accuracy} \n')

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, split="train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, split="val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, split="test")
        

## OK-VQA Dataset
class OkVqaDataset (torchvision.datasets.vision.VisionDataset):
  def __init__(self, root, split, transform=None):
        super(OkVqaDataset, self).__init__(root, transform=transform)

        assert os.path.exists(root), f"Root directory '{root}' does not exist."
        assert split in ['train', 'val', 'test'], f"Unsupported split: {split}"

        self.root = root
        self.split = split

        with open(os.path.join(root, self.split, f'annotations_{self.split}.json'), "r") as f:
            self.annotations = json.load(f)

        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.ToTensor()

  def __getitem__(self, index):
        # Process annotations
        annotation = self.annotations["annotations"][index]
        question = annotation["question"]
        answers = [str(ans["raw_answer"]) for ans in annotation["answers"]]

        # Process image
        image_id = annotation["image_id"]
        image_name = f'okvqa_{self.split}_{str(image_id).zfill(12)}.jpg'
        img_path = os.path.join(self.root, self.split, self.split, image_name)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, question, list(answers) , random.choice(list(answers))

  def __len__(self):
        return len(self.annotations["annotations"])

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
        dataset = OkVqaDataset(root=self.root, split=split, transform=self.transform)
        params = {
            'batch_size': self.batch_size,
            'shuffle': True,
            'num_workers': self.num_workers,
        }
        return data.DataLoader(dataset, **params)

    def val_dataloader(self):
        split = 'val'
        dataset = OkVqaDataset(root=self.root, split=split, transform=self.transform)
        params = {
            'batch_size': self.batch_size,
            'shuffle': False,
            'num_workers': self.num_workers,
        }
        return data.DataLoader(dataset, **params)

    def test_dataloader(self):
        split = 'test'
        dataset = OkVqaDataset(root=self.root, split=split, transform=self.transform)
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
        "--dataset", type=str, default="spatialcoco", choices=["spatialcoco"],
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
        "--iteration", type=int, default=0, help="Iteration number (for Random Search)." 
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
        #"--output_path", type=str, default="/ikerlariak/asalaberria009/trained_models/mm_vsr_models/", help="Output directory for plots and models."
        "--output_path", type=str, default="/gaueko0/users/ietxarri010/ofa_okvqa_finetuning/"
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

    #TODO change default to okvqa!
    if args.dataset == 'spatialcoco': #DEFAULT: spatialcoco
        datamodule = OKVQADataModule(args)

    else:
        raise NotImplementedError
       
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
    #logger = WandbLogger(project=tb_run_name)
    
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
