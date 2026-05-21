import torch
from torch.utils.data import Dataset, DataLoader
import yaml
from PIL import Image
import os
import pandas as pd


class SentimentDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    
    def __len__(self):
        return len(self.data)

    
    def __getitem__(self, index):
        text = str(self.data.text[index])
        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True,
            return_tensors='pt'
        )
        ids = inputs["input_ids"].flatten()
        mask = inputs["attention_mask"].flatten()
        token_type_id = inputs["token_type_ids"].flatten()
        return {
            "ids": ids,
            "mask": mask,
            "token_type_id": token_type_id,
            "targets": torch.tensor(self.data.sentiment[index], dtype=torch.long)
        }
    

class ImageDataset(Dataset):
    def __init__(self, dataframe, image_processor, image_dir):
        self.data = dataframe
        self.image_processor = image_processor
        self.image_dir = image_dir
        self.parts = [f"part{i}" for i in range(1, 7)]  # part1 to part6

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        image_id = row['id']
        
        # Try to find image in each part
        for part in self.parts:
            image_path = os.path.join(self.image_dir, part, f"{image_id}.jpg")
            if os.path.exists(image_path):
                try:
                    image = Image.open(image_path).convert('RGB')
                    pixel_values = self.image_processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
                    return {
                        'pixel_values': pixel_values,
                        'targets': torch.tensor(row['sentiment'], dtype=torch.long)
                    }
                except Exception as e:
                    print(f"Error loading image {image_path}: {e}")
                    continue
        
        # If image not found in any part, return blank image
        print(f"Image {image_id} not found in any part")
        return {
            'pixel_values': torch.zeros((3, 224, 224)),
            'targets': torch.tensor(row['sentiment'], dtype=torch.long)
        }

def data_loader(dataframe, processor, max_len=None, params=None, image_dir=None):
    if image_dir is not None:
        ds = ImageDataset(dataframe, processor, image_dir)
    else:
        ds = SentimentDataset(dataframe, processor, max_len)
    dl = DataLoader(ds, **params)
    return dl

def load_dataframe(file_path):
    """
    Simulate loading a dataset from a CSV file.

    Parameters:
        file_path (str): Path to the dataset file.

    Returns:
        DataFrame: A pandas DataFrame with the loaded data.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded dataset from: {file_path}")
        # df = df.rename(columns={"id": "id"})
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None


def load_experiment_data(alpha_version, dataset_type, experiment_group):
    """
    Load the dataset dynamically based on experiment parameters.

    Parameters:
        alpha_version (int): The alpha version (e.g., 3, 4, 5).
        dataset_type (str): Dataset type, e.g., "percept_dataset" or "gpt4-openai-classify".
        experiment_group (str): Experiment group, e.g., "p5", "p3", "p2plus", "p2neg".

    Returns:
        DataFrame: The loaded dataset.
    """
    # Build the dynamic file path
    file_path = f"data/{dataset_type}/percept_dataset_alpha{alpha_version}_{experiment_group}.csv"

    # Load the dataset
    df = load_dataframe(file_path)
    print(f"Loaded dataset from: {file_path}")
    return df