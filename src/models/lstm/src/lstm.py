
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from builders.model_builder import META_ARCHITECTURE

@META_ARCHITECTURE.register()
class LSTM(nn.Module):
    def __init__(self, config, vocab):
        super(LSTM, self).__init__()
        
        # Create embedding matrix from vocab
        embedding_matrix = self._create_embedding_matrix(vocab, config)
        embedding_tensor = torch.tensor(embedding_matrix, dtype=torch.float)
        self.embed = nn.Embedding.from_pretrained(embedding_tensor, freeze=False)

        self.hidden_dim = config.hidden_dim
        self.bidirectional = config.bidirectional
        self.num_layers = config.num_layers

        self.lstm = nn.LSTM(
            input_size=config.embed_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=self.bidirectional
        )

        output_dim = self.hidden_dim * 2 if self.bidirectional else self.hidden_dim
        
        # Use vocab's class count for output dimension
        if hasattr(vocab, 'get_num_classes'):
            self.num_classes = vocab.get_num_classes()
        elif hasattr(vocab, 'class_labels'):
            self.num_classes = len(vocab.class_labels)
        else:
            self.num_classes = config.polarities_dim
            
        self.dense = nn.Linear(output_dim, self.num_classes)
        self.device = config.device
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(config.dropout if hasattr(config, 'dropout') else 0.1)
        
        # Add loss function - use label_smoothing for better training
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Add d_model attribute for compatibility with scheduler
        self.d_model = config.embed_dim

    def _create_embedding_matrix(self, vocab, config):
        """Create embedding matrix from vocabulary"""
        vocab_size = len(vocab)
        embed_dim = config.embed_dim
        
        # Initialize with Xavier uniform for better convergence
        embedding_matrix = np.random.uniform(-0.1, 0.1, (vocab_size, embed_dim))
        
        # You can add pre-trained embeddings here if available
        # For example, loading Word2Vec, GloVe, or fastText embeddings
        
        return embedding_matrix

    def forward(self, input_ids, labels=None):
        """
        Forward pass that handles both training and inference
        Args:
            input_ids: tensor of shape (batch_size, seq_len)
            labels: tensor of shape (batch_size,) for training, None for inference
        Returns:
            If labels provided: (logits, loss)
            If labels not provided: logits
        """
        text_raw_indices = input_ids  # (batch_size, seq_len)
        batch_size = text_raw_indices.size(0)
        
        # Embedding
        x = self.embed(text_raw_indices)  # (batch_size, seq_len, embed_dim)
        x = self.dropout(x)  # Apply dropout to embeddings

        # Calculate actual lengths for packing
        x_len = torch.sum(text_raw_indices != 0, dim=-1)
        
        # Handle edge case where all sequences have length 0
        if torch.all(x_len == 0):
            # Return zeros for empty sequences
            logits = torch.zeros(batch_size, self.num_classes, device=text_raw_indices.device)
            if labels is not None:
                # Ensure labels are proper indices
                target_labels = self._process_labels(labels)
                loss = self.criterion(logits, target_labels)
                return logits, loss
            return logits
        
        # Sort sequences by length for efficient packing
        x_len_sorted, idx_sort = torch.sort(x_len, descending=True)
        x_sorted = x[idx_sort]
        
        # Filter out sequences with length 0
        valid_mask = x_len_sorted > 0
        if not torch.all(valid_mask):
            x_len_valid = x_len_sorted[valid_mask]
            x_valid = x_sorted[valid_mask]
        else:
            x_len_valid = x_len_sorted
            x_valid = x_sorted

        # Pack sequences for efficient LSTM processing
        packed_input = nn.utils.rnn.pack_padded_sequence(
            x_valid, x_len_valid.cpu(), batch_first=True, enforce_sorted=True
        )
        
        # LSTM forward pass
        packed_output, (ht, ct) = self.lstm(packed_input)

        # Handle the case where we filtered out some sequences
        if not torch.all(valid_mask):
            # Create full hidden state tensor with zeros for invalid sequences
            full_ht = torch.zeros(
                self.num_layers * (2 if self.bidirectional else 1),
                batch_size,
                self.hidden_dim,
                device=ht.device,
                dtype=ht.dtype
            )
            # Fill in valid sequences
            full_ht[:, idx_sort[valid_mask], :] = ht
            ht = full_ht
        else:
            # Reorder to original sequence
            _, idx_unsort = torch.sort(idx_sort)
            ht = ht[:, idx_unsort, :]

        # Extract final hidden state
        if self.bidirectional:
            # Concatenate forward and backward final states
            last_ht = torch.cat((ht[-2], ht[-1]), dim=1)  # (batch, hidden*2)
        else:
            last_ht = ht[-1]  # (batch, hidden)

        # Apply dropout before final layer
        last_ht = self.dropout(last_ht)
        
        # Final classification layer
        logits = self.dense(last_ht)  # (batch, num_classes)
        
        # Calculate loss if labels are provided (training mode)
        if labels is not None:
            target_labels = self._process_labels(labels)
            loss = self.criterion(logits, target_labels)
            return logits, loss
        
        return logits

    def _process_labels(self, labels):
        """Process labels to ensure they are proper class indices"""
        if labels.dim() > 1 and labels.size(1) > 1:
            # One-hot encoded labels - convert to indices
            target_labels = torch.argmax(labels, dim=1)
        else:
            # Already class indices
            target_labels = labels.long()
        
        # Ensure labels are within valid range
        target_labels = torch.clamp(target_labels, 0, self.num_classes - 1)
        return target_labels

    def predict(self, inputs, return_prob=False):
        """Prediction method for inference"""
        device = self.device
        self.eval()
        inputs = inputs.to(device)
        
        with torch.no_grad():
            logits = self.forward(inputs)  # No labels, so only returns logits
            probs = F.softmax(logits, dim=1)
            
            if return_prob:
                return probs
            else:
                preds = torch.argmax(probs, dim=1)
                return preds