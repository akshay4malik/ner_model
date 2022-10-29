
import os
import shutil
os.environ["WANDB_DISABLED"] = "true"

from bentoml import env, artifacts, api, BentoService
from bentoml.adapters import DataframeInput, StringInput
from bentoml.frameworks.transformers import TransformersModelArtifact
from bentoml.frameworks.onnx import OnnxModelArtifact
from ignite_bentoml.adapters import LumeInput, LumeOutput
from ignite_bentoml.api import lumeapi
import pandas as pd
import numpy as np
import uuid
import tempfile
from typing import List, Union
import pickle
from collections import defaultdict
from tqdm import tqdm
import random
from transformers import ElectraTokenizerFast, ElectraModel, Trainer, TrainingArguments, ElectraForTokenClassification
from transformers import pipeline
from transformers.convert_graph_to_onnx import convert
import torch
import onnxruntime as rt
import pyarrow as pa
from transformers import DataCollatorForTokenClassification
from datasets.dataset_dict import Dataset, DatasetDict
from datasets import load_dataset, load_metric
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from pathlib import Path
from sklearn.model_selection import train_test_split
import datasets
from datasets import ClassLabel

from ignite_lume.lume import Lume, LumeElement, LumeDataset
from ignite_lume.lume_types import LumeText

from torch.utils.data import TensorDataset, DataLoader, SequentialSampler


# # If there's a GPU available...
# if torch.cuda.is_available():    

#     # Tell PyTorch to use the GPU.    
#     device = torch.device("cuda")

#     print('There are %d GPU(s) available.' % torch.cuda.device_count())

#     print('We will use the GPU:', torch.cuda.get_device_name(0))

# # If not...
# else:
#     print('No GPU available, using the CPU instead.')
#     device = torch.device("cpu")
# @env(
#    additional_files=["./seqeval_files/seqeval.json", "./seqeval_files/seqeval.py"]
# )
label_all_tokens = True

@artifacts([TransformersModelArtifact('ner_model'), OnnxModelArtifact('onnx_model', backend='onnxruntime')])
class NERModelTransformers(BentoService): 
    def get_entity_context_indices(self, pred_dict_list):
        """
        Function to get predicted entities information
        """
        entity_details = []
        for i in range(len(pred_dict_list)):
            if pred_dict_list[i]['entity'] != 'LABEL_0':
                entity_details.append({"start": pred_dict_list[i]['start'], "end": pred_dict_list[i]['end'], "score": pred_dict_list[i]["score"]})

        return entity_details
    
    def get_distinct_entities(self, entity_indices):
        
        """"
        Function to get distinct entities
        """
        distinct_entities = []
        entity1 = []
        for i in range(len(entity_indices)):
            if i == 0:
                entity1.append(entity_indices[i])
            else:
                if (entity_indices[i]['start'] - entity_indices[i - 1]['end']) < 2:
                    entity1.extend([entity_indices[i]])
                else:
                    distinct_entities.append(entity1)
                    entity1 = []
                    entity1.append(entity_indices[i])
        distinct_entities.append(entity1)

        # get entity span
        spans = []
        scores = []

        for item in distinct_entities:
            spans.append([item[0]['start'], item[-1]['end']])
            scores.append(np.mean([i['score'] for i in item]))
        return spans, scores
    
    def single_prediction(self, context:str):
        """
        Fucntion to predict using context
        """
        model =  self.artifacts.ner_model.get("model")
        tokenizer =  self.artifacts.ner_model.get("tokenizer")
        # Create pipeline object
        pipeline_obj = pipeline("token-classification", model=model, tokenizer=tokenizer)
        pipeline_results = pipeline_obj(context)
        ner_results = self.get_entity_context_indices(pipeline_results)
        ent, score = self.get_distinct_entities(ner_results)
        output_entities = []
        for i,j in zip(ent,score):
            output_entities.append({"text":context[i[0]:i[-1]],
                                    "start_end_ind": i,
                                    "score":j})
        return output_entities
        
    
    def predict_using_pipeline(self, lume:Lume, element_type:str = "ner_input_elm", output_element_type:str = "ner_predicted_elm"):
        """
        Fuction to predcit entities using pipeline for lumes
        """
       # lm_data = Lume.load(lume)
        els = [i for i in lume.elements if i.element_type == element_type]
        text_list = []
        start_end_ind = []
        context_list = []
        for el in els:
            start, end = el.attributes["__attr__start_index"], el.attributes["__attr__end_index"]
            context_list.append(lume.data[start:end])       

        model =  self.artifacts.ner_model.get("model")
        tokenizer =  self.artifacts.ner_model.get("tokenizer")
        # Create pipeline object
        pipeline_obj = pipeline("token-classification", model=model, tokenizer=tokenizer)
        new_elem = []
        for context in context_list:
            pipeline_results = pipeline_obj(context)
            ner_results = self.get_entity_context_indices(pipeline_results)
            ent, score = self.get_distinct_entities(ner_results)
            output_entities = []
            for i in ent:
                output_entities.append(context[i[0]:i[-1]])
            
            attrs = {"__attr__entities": output_entities,
                     "__attr__score": score,
                     "__attr__indices": ent
                     }
            new_elem.append(LumeElement(element_type=output_element_type, attributes=attrs))



        return lume.with_more_elements(new_elem)
    

    
    def get_all_span_with_entity_name(self, listOfSpans, context_len, name):
        """
        Function to extract entities spans
        """
        linear_list = [0]
        for i in listOfSpans:
            linear_list.extend(i)
        linear_list.extend([context_len])
        span_dict_list = []
        for i in range (len(linear_list)-1):
            span = [linear_list[i], linear_list[i+1]]
            if span in listOfSpans:
                span_dict_list.append({"span":span, "ent_type":name})
            else:
                span_dict_list.append({"span":span, "ent_type":"O"})
        return span_dict_list
    
    def create_dataset(self, lume, element_to_train):
        """
        Fucntion to create the dataset from lumes. This generates context randomly keeping the
        entities in context and generate labels as per BIO encoding. 
        Args:
            lume: Lume with required elements
            element_to_train: List of elements which contains Annotated Named Entities
        Returns:
            Returns a dictionary containing list of tokens and labels
            
        """
        tokenizer = self.artifacts.ner_model.get("tokenizer")
        required_elements = [[i.attributes['__attr__start_index'],i.attributes['__attr__end_index']] for i in lume.elements if i.element_type in element_to_train]
        if required_elements != []:
            unique_elements = []
            for i in required_elements:
                if i not in unique_elements:
                    unique_elements.append(i)
            unique_elements = sorted(unique_elements)
            all_start_end_indices = []

            for i in unique_elements:
                all_start_end_indices.extend(i)
            start, end = min(all_start_end_indices), max(all_start_end_indices)

            txt = lume.data[start:end]
            count = 0
            while len(tokenizer(txt).input_ids) < 500:
                # subsequently increase the number of words and take care not to cross the zero or max len limit
                start, end = max(0, start - random.randint(0,50)), min(len(lume.data), end + random.randint(0,50))
                txt = lume.data[start:end]
                count += 1
                if count > 100:
                    break
                    
            listOfSpans = []
            for i in unique_elements:
                start_span = i[0] - start
                len_span = i[-1] - i[0]
                listOfSpans.append([start_span, start_span + len_span])
                
            entity_name = element_to_train[0].split("__")[-1]
            span_dict_list = self.get_all_span_with_entity_name(listOfSpans, len(txt), entity_name)
            tokens = []
            ner_tags = []
            id_ = uuid.uuid4().hex

            for item in span_dict_list:
                if item['ent_type'] == "O":
                    token_list = txt[item['span'][0]:item['span'][1]].split()
                    ner_tag_list = ["O"]*len(token_list)
                    tokens.extend(token_list)
                    ner_tags.extend(ner_tag_list)
                else:
                    token_list = txt[item['span'][0]:item['span'][1]].split()
                    ner_tag_list = []
                    if len(token_list) > 0:
                        for i in range (len(token_list)):
                            if i == 0:
                                ner_tag_list.append(f"B-{entity_name}")
                            else:
                                ner_tag_list.append(f"I-{entity_name}")

                    tokens.extend(token_list)
                    ner_tags.extend(ner_tag_list)
 
            squad_json = {
                        "tokens": tokens,
                        "ner_tags": ner_tags,
                        "id": id_
                    }          
            return squad_json
        else:
            return {}
        
    def map_df_rows(self, df):
        """
        Function to map DataFrame Label row with integers, The dictionary given in below
        function can be updated as per the requirments
        Args:
            df: Input DataFrame for which we need to transform ner_tags label
        Returns:
            List of mapped integers with labels
        """
        updated_rows = []
        # The following map dict can be updated if more annotations comes in
        map_df_rows_dict = {"O":0, "B-contracting_party":1, "I-contracting_party":2}
        for row in list(df['ner_tags']):
            updated_rows.append(list(map(map_df_rows_dict.get, row)))

        return updated_rows
    
    def tokenize_and_align_labels(self, examples):
        """
        Function for tokenizing and aligning labels
        """
        tokenizer = self.artifacts.ner_model.get("tokenizer")
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True,max_length=512, is_split_into_words=True)

        labels = []
        task = "ner"
        for i, label in enumerate(examples[f"{task}_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(label[word_idx] if label_all_tokens else -100)
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        tokenized_inputs["tokens"] = examples["tokens"]
        tokenized_inputs["id"] = examples["id"]
        return tokenized_inputs

    def compute_metrics(self, p):
        """
        Function to compute different metrices for model training
        """
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        label_list = ['B-contracting_party', 'I-contracting_party', 'O']
        metric = load_metric("seqeval_files/seqeval.py")
        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        results = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }


    @api(api_name="train_model", input=LumeInput(), output=LumeOutput())
    def train(
        self,
        lume_dataset: LumeDataset,
        output_path: str,
        element_type: List,
        validation_split: Union[float,list]=0.05,
        output_version: str="finetuned_model",
        **kwargs
        ) -> None:
        
        """
        Fine-tunes a cross-encoder style model based on Google's Electra transformer architecture
        for NER (Token Classification) task.
        
        The fine-tuned model will be saved as a new Bento at the path specified.
        
        Args:
            lume_dataset (LumeDataset): The input LumeDataset contain the Lumes with appropriate training data
            output_path (str): Where to save the fine-tuned model as a new Bento package. If set to be the same directory
                               as the starting Bento, after training only the artifact files and `bentoml.yml` metadata
                               files will be updated.
            element_type (Union[str,list]): The LumeElement type(s) containing the training data
            validation_split (Union[float,list]): Either a fraction (0-1) for random splitting, or a list of
                                                  Lume names used to created the the validation data.
                                                  Splitting by Lume names is recommended when possible at it
                                                  provides the most control over the content of the validation set.
                                                  (default: 0.1)
            output_version (str): The version written in the metadata for the newly created Bento. (default: "finetuned_model")
                                The user is encouraged to follow semantic versioning principles.
            kwargs: Any other keyword arguments to pass to the HuggingFace Trainer object.
            
        """

        
        # Prepare the dataset 
        data = []
        for lume in lume_dataset.lumes:
            res = self.create_dataset(lume, element_type)
            if res != {}:
                data.append(res)
        
        data_df = pd.DataFrame(data)
        
        # Creating the DataFrame from the dataset lists
        train_data_df,val_data_df  = train_test_split(data_df,test_size=validation_split, shuffle=True)
        
        dataset_info = datasets.DatasetInfo(
            description="_DESCRIPTION",
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=sorted(list(("B-contracting_party","I-contracting_party", "O")))
                        )
                    )
                }
            ),
            supervised_keys=None,
            homepage="",
            citation="_CITATION",
        )
        
        # Map the Labels with integers
        val_data_df['ner_tags'] = self.map_df_rows(val_data_df)
        train_data_df['ner_tags'] = self.map_df_rows(train_data_df)
        train_data = Dataset.from_pandas(train_data_df,info= dataset_info)
        val_data = Dataset.from_pandas(val_data_df,info= dataset_info)
        
        # Create Dataset Dictionary
        dataset_dict = DatasetDict({"train": train_data, 'val': val_data})
        task = "ner"
        label_all_tokens = True
        label_list = dataset_dict["train"].features["ner_tags"].feature.names
        # Tokenize the dataset context        
        tokenized_datasets = dataset_dict.map(self.tokenize_and_align_labels, batched=True, remove_columns = dataset_dict["train"].column_names)
        # Define Model
        model =  self.artifacts.ner_model.get("model")
        tokenizer = self.artifacts.ner_model.get("tokenizer")
        
        model_name = "electra_small"
        data_collator = DataCollatorForTokenClassification(tokenizer)    
    
        with tempfile.TemporaryDirectory() as tmp_dir:
        # Define training arguments for HF trainer
            training_args = TrainingArguments(f"{model_name}-finetuned-{task}",
              #  output_dir=tmp_dir,
                learning_rate=kwargs.get('learning_rate', 2e-5),
                per_device_train_batch_size=kwargs.get('per_device_train_batch_size',8),
                per_device_eval_batch_size=kwargs.get('per_device_eval_batch_size', 8),
                num_train_epochs=kwargs.get('num_train_epochs', 1),
                logging_dir=os.path.join(tmp_dir, "logs"),
                logging_steps=kwargs.get('logging_steps', max(1, len(train_data)//8//4)),
                save_steps=kwargs.get('save_steps', max(1, len(train_data)//48//4)),
                save_total_limit=2,              
                evaluation_strategy="epoch",
                eval_steps=kwargs.get('eval_steps', max(1, len(train_data)//8//4)),
                seed=kwargs.get('seed', 42)
            )
            

            # Allow user to set/override any other training arguments
            for key in kwargs.keys():
                if key in training_args.__dict__:
                    training_args.__setattr__(key, kwargs[key])


            trainer = Trainer(
                model=self.artifacts.ner_model.get("model"),                         # the instantiated 🤗 Transformers model to be trained
                args=training_args,                  # training arguments, defined above
                train_dataset=tokenized_datasets["train"],           # training dataset
                eval_dataset=tokenized_datasets["val"],            # evaluation dataset,
                data_collator=data_collator,
                tokenizer=tokenizer,
                compute_metrics=self.compute_metrics
            )

            # Start the training
            trainer.train()

            # Save model at the end of training
            model_save_path = os.path.join(tmp_dir, "trained_model")
            if not os.path.exists(model_save_path):
                os.mkdir(model_save_path)

            # Save model
            trainer.save_model(model_save_path)
            self.artifacts.ner_model.get("tokenizer").save_pretrained(model_save_path)
            print("Training complete!")

            # Export ONNX version of model, and pack new Bento
            with tempfile.TemporaryDirectory() as tmp_dir_onnx:
                convert(framework="pt", model=model_save_path, output=Path(os.path.join(tmp_dir_onnx, "onnx_model.onnx")), opset=11, pipeline_name="ner")

                # save bento model for the newly trained onnx and transformer models
                new_transformer_model = ElectraForTokenClassification.from_pretrained(model_save_path)
                new_tokenizer = ElectraTokenizerFast.from_pretrained(model_save_path, use_fast=True)
                artifact = {"model": new_transformer_model, "tokenizer": new_tokenizer}

                # Pack and resave the model
                self.pack("ner_model", artifact)
                self.pack("onnx_model", os.path.join(tmp_dir_onnx, "onnx_model.onnx"))
                self.save_to_dir(output_path)
        return None

