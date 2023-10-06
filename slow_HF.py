import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForMaskedLM, AdamW
from datasets import load_dataset
from transformers import BertTokenizer
from transformers import BertForMaskedLM, AdamW


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.attn_masks = []

        for text in texts:
            encoding = tokenizer(text, max_length=max_length, padding="max_length", truncation=True, return_tensors='pt')
            self.input_ids.append(encoding['input_ids'])
            self.attn_masks.append(encoding['attention_mask'])
        
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]

if __name__ == "__main__":
    # Load dataset using Hugging Face
    wikitext = load_dataset('wikitext', 'wikitext-103-raw-v1')

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-tiny")

    # Example usage of the TextDataset and DataLoader
    train_texts = wikitext['train']['text'][:2000]  # Subset for demonstration
    val_texts = wikitext['validation']['text'][:2000]  # Subset for demonstration
    train_dataset = TextDataset(train_texts, tokenizer, max_length=128)
    val_dataset = TextDataset(val_texts, tokenizer, max_length=128)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True)

    # Initialize model
    model = BertForMaskedLM.from_pretrained("prajjwal1/bert-tiny")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device="mps"
    model.to(device)

    # Loss and optimizer
    optimizer = AdamW(model.parameters(), lr=1e-5)

    # Training loop
    num_epochs = 3  # Specify the number of epochs

    for epoch in range(num_epochs):
        total_train_loss = 0
        total_val_loss = 0
        
        for input_ids_batch, attn_masks_batch in train_loader:
            # Move to device
            input_ids_batch = input_ids_batch.squeeze().to(device)
            attn_masks_batch = attn_masks_batch.squeeze().to(device)
            
            # Forward pass and calculate loss
            outputs = model(input_ids=input_ids_batch, attention_mask=attn_masks_batch, labels=input_ids_batch)
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        for input_ids_batch, attn_masks_batch in val_loader:
            input_ids_batch = input_ids_batch.squeeze().to(device)
            attn_masks_batch = attn_masks_batch.squeeze().to(device)
            outputs = model(input_ids=input_ids_batch, attention_mask=attn_masks_batch, labels=input_ids_batch)
            loss = outputs.loss
            total_val_loss += loss.item()

        print(f'Epoch {epoch+1}, Train Loss: {total_train_loss/len(train_loader)}, Val Loss: {total_val_loss/len(val_loader)}')

    input_ids = tokenizer.encode("Hello habibi, my name is", return_tensors='pt').to(device)
    output = model.generate(input_ids, max_length=50, num_beams=5, temperature=1.5)
    print(tokenizer.decode(output[0], skip_special_tokens=True))
