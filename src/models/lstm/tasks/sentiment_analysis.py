# # sentiment_task.py

# from torch.utils.data import DataLoader
# import os
# import torch
# from tqdm import tqdm
# import json
# from builders.dataset_builder import build_dataset
# from builders.task_builder import META_TASK
# from tasks.base_task import BaseTask
# from datasets import collate_fn
# from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
# import numpy as np


# @META_TASK.register()
# class SentimentAnalysisTask(BaseTask):
#     def __init__(self, config):
#         super().__init__(config)

#     def configuring_hyperparameters(self, config):
#         self.epoch = 0
#         self.score = config.training.score
#         self.learning_rate = config.training.learning_rate
#         self.patience = config.training.patience
#         self.warmup = config.training.warmup

#     def load_datasets(self, config):
#         # Build datasets with shared vocab
#         self.train_dataset = build_dataset(config.train, self.vocab)
#         self.dev_dataset = build_dataset(config.dev, self.vocab)
#         self.test_dataset = build_dataset(config.test, self.vocab)
        
#         # Verify label consistency across datasets
#         self._verify_label_consistency()

#     def _verify_label_consistency(self):
#         """Verify that all datasets have consistent labels"""
#         datasets = [self.train_dataset, self.dev_dataset, self.test_dataset]
#         dataset_names = ['train', 'dev', 'test']
        
#         # Use vocab's class labels as the reference
#         if hasattr(self.vocab, 'class_labels'):
#             reference_labels = sorted(self.vocab.class_labels)
#             self.class_labels = reference_labels
#             self.label_to_idx = self.vocab.class_to_idx
#             self.idx_to_label = self.vocab.idx_to_class
#             self.num_classes = len(reference_labels)
            
#             self.logger.info(f"Using vocab class labels: {reference_labels}")
#             self.logger.info(f"Number of classes: {self.num_classes}")
#             self.logger.info(f"Label to index mapping: {self.label_to_idx}")
#         else:
#             # Fallback to dataset labels
#             reference_labels = None
#             for i, (dataset, name) in enumerate(zip(datasets, dataset_names)):
#                 if hasattr(dataset, 'labels'):
#                     current_labels = sorted(dataset.labels)
#                     if reference_labels is None:
#                         reference_labels = current_labels
#                         self.logger.info(f"Reference labels from {name}: {reference_labels}")
#                     elif current_labels != reference_labels:
#                         self.logger.warning(f"Label mismatch in {name}: {current_labels} vs {reference_labels}")
                        
#             # Store label mappings for evaluation
#             if hasattr(self.train_dataset, 'label_to_idx'):
#                 self.label_to_idx = self.train_dataset.label_to_idx
#                 self.idx_to_label = self.train_dataset.idx_to_label
#                 self.num_classes = len(self.label_to_idx)
#                 self.logger.info(f"Number of classes: {self.num_classes}")
#                 self.logger.info(f"Label to index mapping: {self.label_to_idx}")

#     def create_dataloaders(self, config):
#         self.train_dataloader = DataLoader(
#             dataset=self.train_dataset,
#             batch_size=config.dataset.batch_size,
#             shuffle=True,
#             num_workers=config.dataset.num_workers,
#             collate_fn=collate_fn
#         )
#         self.dev_dataloader = DataLoader(
#             dataset=self.dev_dataset,
#             batch_size=config.dataset.batch_size,
#             shuffle=False,
#             num_workers=config.dataset.num_workers,
#             collate_fn=collate_fn
#         )
#         self.test_dataloader = DataLoader(
#             dataset=self.test_dataset,
#             batch_size=config.dataset.batch_size,
#             shuffle=False,
#             num_workers=config.dataset.num_workers,
#             collate_fn=collate_fn
#         )
    
#     def get_vocab(self): 
#         return self.vocab

#     def train(self):
#         self.model.train()

#         running_loss = 0.0
#         correct_predictions = 0
#         total_predictions = 0
        
#         with tqdm(desc='Epoch %d - Training' % (self.epoch+1), unit='it', total=len(self.train_dataloader)) as pbar:
#             for it, items in enumerate(self.train_dataloader):
#                 items = items.to(self.device)
#                 input_ids = items.input_ids
#                 labels = items.labels
                
#                 # Ensure labels are proper class indices
#                 target_labels = self._process_labels(labels)

#                 # Forward pass - model handles loss calculation internally
#                 logits, loss = self.model(input_ids, target_labels)
                
#                 # Calculate accuracy for monitoring
#                 predictions = torch.argmax(logits, dim=-1)
#                 correct_predictions += (predictions == target_labels).sum().item()
#                 total_predictions += target_labels.size(0)
                
#                 # Backward pass
#                 self.optim.zero_grad()
#                 loss.backward()
                
#                 # Gradient clipping for stability
#                 torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
#                 self.optim.step()
#                 self.scheduler.step()  # Update learning rate

#                 running_loss += loss.item()
                
#                 # Update progress bar
#                 current_acc = correct_predictions / total_predictions
#                 pbar.set_postfix(
#                     loss=f"{running_loss / (it + 1):.4f}",
#                     acc=f"{current_acc:.4f}"
#                 )
#                 pbar.update()
        
#         # Log training statistics
#         final_acc = correct_predictions / total_predictions
#         avg_loss = running_loss / len(self.train_dataloader)
#         self.logger.info(f"Training - Loss: {avg_loss:.4f}, Accuracy: {final_acc:.4f}")

#     def _process_labels(self, labels):
#         """Process labels to ensure they are proper class indices"""
#         if labels.dim() > 1 and labels.size(-1) > 1:
#             # One-hot encoded labels - convert to indices
#             target_labels = torch.argmax(labels, dim=-1)
#         else:
#             # Already class indices
#             target_labels = labels.long()
        
#         # Clamp to valid range
#         target_labels = torch.clamp(target_labels, 0, self.num_classes - 1)
#         return target_labels

#     def evaluate_metrics(self, dataloader: DataLoader) -> dict:
#         self.model.eval()
#         preds = []
#         trues = []
#         total_loss = 0.0
        
#         with tqdm(desc='Epoch %d - Evaluating' % (self.epoch+1), unit='it', total=len(dataloader)) as pbar:
#             for items in dataloader:
#                 items = items.to(self.device)
#                 input_ids = items.input_ids
#                 labels = items.labels
                
#                 # Process labels consistently
#                 true_labels = self._process_labels(labels)
                
#                 with torch.no_grad():
#                     # Forward pass
#                     logits, loss = self.model(input_ids, true_labels)
#                     total_loss += loss.item()
                    
#                     # Get predictions
#                     pred_labels = torch.argmax(logits, dim=-1)
                    
#                     # Collect predictions and true labels
#                     preds.extend(pred_labels.cpu().tolist())
#                     trues.extend(true_labels.cpu().tolist())

#                 pbar.update()

#         # Log some sample predictions for debugging
#         self.logger.info("Sample predictions (first 10):")
#         for i in range(min(10, len(preds))):
#             pred_label = self.idx_to_label.get(preds[i], f"Unknown({preds[i]})")
#             true_label = self.idx_to_label.get(trues[i], f"Unknown({trues[i]})")
#             self.logger.info(f"  Sample {i}: Pred={pred_label}, True={true_label}")

#         # Compute classification metrics
#         self.logger.info("Computing classification metrics")
        
#         try:
#             # Handle case where some classes might not be predicted
#             unique_labels = sorted(set(trues + preds))
            
#             scores = {
#                 "accuracy": accuracy_score(trues, preds),
#                 "f1_macro": f1_score(trues, preds, average="macro", zero_division=0),
#                 "f1_micro": f1_score(trues, preds, average="micro", zero_division=0),
#                 "f1_weighted": f1_score(trues, preds, average="weighted", zero_division=0),
#                 "precision_macro": precision_score(trues, preds, average="macro", zero_division=0),
#                 "recall_macro": recall_score(trues, preds, average="macro", zero_division=0),
#                 "avg_loss": total_loss / len(dataloader)
#             }
            
#             # Generate detailed classification report
#             target_names = [self.idx_to_label.get(i, f"Class_{i}") for i in range(self.num_classes)]
            
#             # Only include labels that exist in the data
#             labels_in_data = sorted(set(trues + preds))
#             target_names_filtered = [self.idx_to_label.get(i, f"Class_{i}") for i in labels_in_data]
            
#             classification_rep = classification_report(
#                 trues, preds, 
#                 labels=labels_in_data,
#                 target_names=target_names_filtered,
#                 digits=4,
#                 zero_division=0
#             )
#             scores["classification_report"] = classification_rep
            
#             # Print classification report
#             print("\nDetailed Classification Report:")
#             print(classification_rep)
            
#             # Print label distribution
#             print("\nLabel Distribution:")
#             from collections import Counter
#             true_counter = Counter(trues)
#             pred_counter = Counter(preds)
            
#         except Exception as e:
#             self.logger.error(f"Error computing metrics: {e}")
#             import traceback
#             traceback.print_exc()
            
#             scores = {
#                 "accuracy": 0.0,
#                 "f1_macro": 0.0,
#                 "f1_micro": 0.0,
#                 "f1_weighted": 0.0,
#                 "precision_macro": 0.0,
#                 "recall_macro": 0.0,
#                 "avg_loss": total_loss / len(dataloader),
#                 "classification_report": "Error computing report"
#             }

#         return scores, (preds, trues)


#     def get_predictions(self, dataloader=None, calculate_accuracy=True):
#         """Get predictions in the exact required format - grouped by original text"""
#         import numpy as np
#         from collections import defaultdict
        
#         if dataloader is None:
#             dataloader = self.test_dataloader
                
#         self.model.eval()
#         all_true_labels = []
#         all_pred_labels = []
        
#         # Dictionary to group spans by original text
#         text_predictions = defaultdict(list)
                
#         with torch.no_grad():
#             for items in tqdm(dataloader, desc="Generating predictions"):
#                 items = items.to(self.device)
#                 input_ids = items.input_ids
                
#                 # Get model predictions
#                 logits = self.model(input_ids)
#                 pred_classes = torch.argmax(logits, dim=-1).cpu().numpy()
                
#                 # Get batch information
#                 span_texts = items.original_text if hasattr(items, 'original_text') else [f"text_{i}" for i in range(len(input_ids))]
#                 original_texts = items.full_text if hasattr(items, 'full_text') else span_texts
#                 span_positions = items.span_positions if hasattr(items, 'span_positions') else [(0, len(text)) for text in span_texts]
                        
#                 # Process each item in the batch
#                 for i in range(len(input_ids)):
#                     pred_idx = pred_classes[i]
#                     pred_label = self.idx_to_label.get(pred_idx, "Khác#Tích cực")
                    
#                     # Get span information
#                     span_text = span_texts[i] if i < len(span_texts) else ""
#                     original_text = original_texts[i] if i < len(original_texts) else span_text
                    
#                     # Get span position in original text
#                     if hasattr(items, 'span_positions') and i < len(items.span_positions):
#                         start_pos, end_pos = items.span_positions[i]
#                     else:
#                         # Try to find the span in the original text
#                         start_pos = original_text.find(span_text)
#                         if start_pos == -1:
#                             start_pos = 0
#                         end_pos = start_pos + len(span_text)
                    
#                     # Store for accuracy calculation
#                     if hasattr(items, 'labels') and calculate_accuracy:
#                         try:
#                             # Fix: Handle multi-dimensional labels properly
#                             if hasattr(items.labels, 'cpu'):
#                                 label_tensor = items.labels[i].cpu()
                                
#                                 # Check if labels are one-hot encoded or multi-dimensional
#                                 if label_tensor.dim() > 0 and label_tensor.numel() > 1:
#                                     # One-hot encoded or multi-class - get the argmax
#                                     true_label = torch.argmax(label_tensor, dim=-1).item()
#                                 else:
#                                     # Single scalar value
#                                     true_label = label_tensor.item()
#                             else:
#                                 # Handle numpy arrays or lists
#                                 label_data = items.labels[i]
#                                 if hasattr(label_data, 'shape') and len(label_data.shape) > 0 and label_data.shape[0] > 1:
#                                     true_label = np.argmax(label_data)
#                                 else:
#                                     true_label = int(label_data)
                            
#                             all_pred_labels.append(int(pred_idx))
#                             all_true_labels.append(int(true_label))
                            
#                         except Exception as e:
#                             print(f"Error processing labels at index {i}: {e}")
#                             print(f"Label shape: {items.labels[i].shape if hasattr(items.labels[i], 'shape') else 'No shape'}")
#                             print(f"Label value: {items.labels[i]}")
#                             calculate_accuracy = False
                    
#                     # Add span to the corresponding original text
#                     text_predictions[original_text].append([start_pos, end_pos, span_text, pred_label])
        
#         # Convert to final format
#         predictions = []
#         for original_text, labels in text_predictions.items():
#             predictions.append({
#                 "text": original_text,
#                 "labels": labels
#             })
        
#         # Calculate accuracy if ground truth is available
#         accuracy = None
#         if calculate_accuracy and all_true_labels and all_pred_labels:
#             true_array = np.array(all_true_labels)
#             pred_array = np.array(all_pred_labels)
#             correct = np.sum(true_array == pred_array)
#             accuracy = correct / len(all_true_labels)
#             print(f"Test Accuracy: {accuracy:.4f} ({correct}/{len(all_true_labels)})")
        
#         # Save predictions
#         save_path = os.path.join(self.checkpoint_path, "predictions.json")
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#         with open(save_path, "w", encoding="utf-8") as f:
#             json.dump(predictions, f, ensure_ascii=False, indent=2)
        
#         print(f"Predictions saved to: {save_path}")
#         print(f"Total predictions: {len(predictions)} texts with spans")

#         scores, _ = self.evaluate_metrics(self.test_dataloader)
        
#         return predictions, accuracy, scores

# sentiment_task.py

from torch.utils.data import DataLoader
import os
import torch
from tqdm import tqdm
import json
from builders.dataset_builder import build_dataset
from builders.task_builder import META_TASK
from tasks.base_task import BaseTask
from datasets import collate_fn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import numpy as np


@META_TASK.register()
class SentimentAnalysisTask(BaseTask):
    def __init__(self, config):
        super().__init__(config)

    def configuring_hyperparameters(self, config):
        self.epoch = 0
        self.score = config.training.score
        self.learning_rate = config.training.learning_rate
        self.patience = config.training.patience
        self.warmup = config.training.warmup

    def load_datasets(self, config):
        # Build datasets with shared vocab
        self.train_dataset = build_dataset(config.train, self.vocab)
        self.dev_dataset = build_dataset(config.dev, self.vocab)
        self.test_dataset = build_dataset(config.test, self.vocab)
        
        # Verify label consistency across datasets
        self._verify_label_consistency()

    def _verify_label_consistency(self):
        """Verify that all datasets have consistent labels"""
        datasets = [self.train_dataset, self.dev_dataset, self.test_dataset]
        dataset_names = ['train', 'dev', 'test']
        
        # Use vocab's class labels as the reference
        if hasattr(self.vocab, 'class_labels'):
            reference_labels = sorted(self.vocab.class_labels)
            self.class_labels = reference_labels
            self.label_to_idx = self.vocab.class_to_idx
            self.idx_to_label = self.vocab.idx_to_class
            self.num_classes = len(reference_labels)
            
            self.logger.info(f"Using vocab class labels: {reference_labels}")
            self.logger.info(f"Number of classes: {self.num_classes}")
            self.logger.info(f"Label to index mapping: {self.label_to_idx}")
        else:
            # Fallback to dataset labels
            reference_labels = None
            for i, (dataset, name) in enumerate(zip(datasets, dataset_names)):
                if hasattr(dataset, 'labels'):
                    current_labels = sorted(dataset.labels)
                    if reference_labels is None:
                        reference_labels = current_labels
                        self.logger.info(f"Reference labels from {name}: {reference_labels}")
                    elif current_labels != reference_labels:
                        self.logger.warning(f"Label mismatch in {name}: {current_labels} vs {reference_labels}")
                        
            # Store label mappings for evaluation
            if hasattr(self.train_dataset, 'label_to_idx'):
                self.label_to_idx = self.train_dataset.label_to_idx
                self.idx_to_label = self.train_dataset.idx_to_label
                self.num_classes = len(self.label_to_idx)
                self.class_labels = list(self.label_to_idx.keys())
                self.logger.info(f"Number of classes: {self.num_classes}")
                self.logger.info(f"Label to index mapping: {self.label_to_idx}")

    def create_dataloaders(self, config):
        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=config.dataset.batch_size,
            shuffle=True,
            num_workers=config.dataset.num_workers,
            collate_fn=collate_fn
        )
        self.dev_dataloader = DataLoader(
            dataset=self.dev_dataset,
            batch_size=config.dataset.batch_size,
            shuffle=False,
            num_workers=config.dataset.num_workers,
            collate_fn=collate_fn
        )
        self.test_dataloader = DataLoader(
            dataset=self.test_dataset,
            batch_size=config.dataset.batch_size,
            shuffle=False,
            num_workers=config.dataset.num_workers,
            collate_fn=collate_fn
        )
    
    def get_vocab(self): 
        return self.vocab

    def train(self):
        self.model.train()

        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with tqdm(desc='Epoch %d - Training' % (self.epoch+1), unit='it', total=len(self.train_dataloader)) as pbar:
            for it, items in enumerate(self.train_dataloader):
                items = items.to(self.device)
                input_ids = items.input_ids
                labels = items.labels
                
                # Ensure labels are proper class indices
                target_labels = self._process_labels(labels)

                # Forward pass - model handles loss calculation internally
                logits, loss = self.model(input_ids, target_labels)
                
                # Calculate accuracy for monitoring
                predictions = torch.argmax(logits, dim=-1)
                correct_predictions += (predictions == target_labels).sum().item()
                total_predictions += target_labels.size(0)
                
                # Backward pass
                self.optim.zero_grad()
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optim.step()
                if hasattr(self, 'scheduler') and self.scheduler is not None:
                    self.scheduler.step()  # Update learning rate

                running_loss += loss.item()
                
                # Update progress bar
                current_acc = correct_predictions / total_predictions
                pbar.set_postfix(
                    loss=f"{running_loss / (it + 1):.4f}",
                    acc=f"{current_acc:.4f}"
                )
                pbar.update()
        
        # Log training statistics
        final_acc = correct_predictions / total_predictions
        avg_loss = running_loss / len(self.train_dataloader)
        self.logger.info(f"Training - Loss: {avg_loss:.4f}, Accuracy: {final_acc:.4f}")

    def _process_labels(self, labels):
        """Process labels to ensure they are proper class indices"""
        if labels.dim() > 1 and labels.size(-1) > 1:
            # One-hot encoded labels - convert to indices
            target_labels = torch.argmax(labels, dim=-1)
        else:
            # Already class indices
            target_labels = labels.long()
        
        # Clamp to valid range
        target_labels = torch.clamp(target_labels, 0, self.num_classes - 1)
        return target_labels

    def evaluate_metrics(self, dataloader: DataLoader) -> dict:
        self.model.eval()
        preds = []
        trues = []
        total_loss = 0.0
        
        with tqdm(desc='Epoch %d - Evaluating' % (self.epoch+1), unit='it', total=len(dataloader)) as pbar:
            for items in dataloader:
                items = items.to(self.device)
                input_ids = items.input_ids
                labels = items.labels
                
                # Process labels consistently
                true_labels = self._process_labels(labels)
                
                with torch.no_grad():
                    # Forward pass
                    logits, loss = self.model(input_ids, true_labels)
                    total_loss += loss.item()
                    
                    # Get predictions
                    pred_labels = torch.argmax(logits, dim=-1)
                    
                    # Collect predictions and true labels
                    preds.extend(pred_labels.cpu().tolist())
                    trues.extend(true_labels.cpu().tolist())

                pbar.update()

        # Log some sample predictions for debugging
        self.logger.info("Sample predictions (first 10):")
        for i in range(min(10, len(preds))):
            pred_label = self.idx_to_label.get(preds[i], f"Unknown({preds[i]})")
            true_label = self.idx_to_label.get(trues[i], f"Unknown({trues[i]})")
            self.logger.info(f"  Sample {i}: Pred={pred_label}, True={true_label}")

        # Compute classification metrics
        self.logger.info("Computing classification metrics")
        
        try:
            # Handle case where some classes might not be predicted
            unique_labels = sorted(set(trues + preds))
            
            scores = {
                "accuracy": accuracy_score(trues, preds),
                "f1_macro": f1_score(trues, preds, average="macro", zero_division=0),
                "f1_micro": f1_score(trues, preds, average="micro", zero_division=0),
                "f1_weighted": f1_score(trues, preds, average="weighted", zero_division=0),
                "precision_macro": precision_score(trues, preds, average="macro", zero_division=0),
                "recall_macro": recall_score(trues, preds, average="macro", zero_division=0),
                "avg_loss": total_loss / len(dataloader)
            }
            
            # Generate detailed classification report
            target_names = [self.idx_to_label.get(i, f"Class_{i}") for i in range(self.num_classes)]
            
            # Only include labels that exist in the data
            labels_in_data = sorted(set(trues + preds))
            target_names_filtered = [self.idx_to_label.get(i, f"Class_{i}") for i in labels_in_data]
            
            classification_rep = classification_report(
                trues, preds, 
                labels=labels_in_data,
                target_names=target_names_filtered,
                digits=4,
                zero_division=0
            )
            scores["classification_report"] = classification_rep
            
            # Print classification report
            print("\nDetailed Classification Report:")
            print(classification_rep)
            
            # Print label distribution
            print("\nLabel Distribution:")
            from collections import Counter
            true_counter = Counter(trues)
            pred_counter = Counter(preds)
            
            print("True labels:", dict(true_counter))
            print("Predicted labels:", dict(pred_counter))
            
        except Exception as e:
            self.logger.error(f"Error computing metrics: {e}")
            import traceback
            traceback.print_exc()
            
            scores = {
                "accuracy": 0.0,
                "f1_macro": 0.0,
                "f1_micro": 0.0,
                "f1_weighted": 0.0,
                "precision_macro": 0.0,
                "recall_macro": 0.0,
                "avg_loss": total_loss / len(dataloader),
                "classification_report": "Error computing report"
            }

        return scores, (preds, trues)

    def get_predictions(self, dataloader=None, calculate_accuracy=True):
        """Get predictions in the exact required format - grouped by original text"""
        import numpy as np
        from collections import defaultdict
        
        if dataloader is None:
            dataloader = self.test_dataloader
                
        self.model.eval()
        all_true_labels = []
        all_pred_labels = []
        
        # Dictionary to group spans by original text
        text_predictions = defaultdict(list)
                
        with torch.no_grad():
            for items in tqdm(dataloader, desc="Generating predictions"):
                items = items.to(self.device)
                input_ids = items.input_ids
                
                # Get model predictions - handle different model outputs
                model_output = self.model(input_ids)
                if isinstance(model_output, tuple):
                    logits = model_output[0]
                else:
                    logits = model_output
                
                pred_classes = torch.argmax(logits, dim=-1).cpu().numpy()
                
                # Get batch information with safe attribute access
                span_texts = getattr(items, 'original_text', [f"text_{i}" for i in range(len(input_ids))])
                original_texts = getattr(items, 'full_text', span_texts)
                span_positions = getattr(items, 'span_positions', [(0, len(text)) for text in span_texts])
                        
                # Process each item in the batch
                for i in range(len(input_ids)):
                    pred_idx = pred_classes[i]
                    pred_label = self.idx_to_label.get(pred_idx, "Unknown")
                    
                    # Get span information with bounds checking
                    span_text = span_texts[i] if i < len(span_texts) else ""
                    original_text = original_texts[i] if i < len(original_texts) else span_text
                    
                    # Get span position in original text
                    if i < len(span_positions):
                        start_pos, end_pos = span_positions[i]
                    else:
                        # Try to find the span in the original text
                        start_pos = original_text.find(span_text) if span_text else 0
                        if start_pos == -1:
                            start_pos = 0
                        end_pos = start_pos + len(span_text)
                    
                    # Store for accuracy calculation
                    if hasattr(items, 'labels') and calculate_accuracy:
                        try:
                            # Handle different label formats
                            label_data = items.labels[i]
                            
                            if hasattr(label_data, 'cpu'):
                                label_tensor = label_data.cpu()
                                
                                # Check dimensions and convert appropriately
                                if label_tensor.dim() > 0 and label_tensor.numel() > 1:
                                    # Multi-dimensional - get argmax
                                    true_label = torch.argmax(label_tensor, dim=-1).item()
                                else:
                                    # Scalar value
                                    true_label = label_tensor.item()
                            else:
                                # Handle numpy arrays or lists
                                if hasattr(label_data, 'shape') and len(label_data.shape) > 0 and label_data.shape[0] > 1:
                                    true_label = np.argmax(label_data)
                                else:
                                    true_label = int(label_data)
                            
                            all_pred_labels.append(int(pred_idx))
                            all_true_labels.append(int(true_label))
                            
                        except Exception as e:
                            self.logger.warning(f"Error processing labels at index {i}: {e}")
                            calculate_accuracy = False
                    
                    # Add span to the corresponding original text
                    text_predictions[original_text].append([start_pos, end_pos, span_text, pred_label])
        
        # Convert to final format
        predictions = []
        for original_text, labels in text_predictions.items():
            predictions.append({
                "text": original_text,
                "labels": labels
            })
        
        # Calculate accuracy if ground truth is available
        accuracy = None
        if calculate_accuracy and all_true_labels and all_pred_labels:
            try:
                true_array = np.array(all_true_labels)
                pred_array = np.array(all_pred_labels)
                correct = np.sum(true_array == pred_array)
                accuracy = correct / len(all_true_labels)
                print(f"Test Accuracy: {accuracy:.4f} ({correct}/{len(all_true_labels)})")
            except Exception as e:
                self.logger.warning(f"Error calculating accuracy: {e}")
                accuracy = None
        
        # Save predictions
        if hasattr(self, 'checkpoint_path'):
            save_path = os.path.join(self.checkpoint_path, "predictions.json")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            try:
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(predictions, f, ensure_ascii=False, indent=2)
                print(f"Predictions saved to: {save_path}")
            except Exception as e:
                self.logger.warning(f"Could not save predictions: {e}")
        
        print(f"Total predictions: {len(predictions)} texts with spans")

        # Get evaluation scores
        try:
            scores, _ = self.evaluate_metrics(self.test_dataloader)
        except Exception as e:
            self.logger.warning(f"Error evaluating metrics: {e}")
            scores = {}
        
        return predictions, accuracy, scores