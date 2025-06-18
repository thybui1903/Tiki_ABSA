# # -*- coding: utf-8 -*-
# # Vietnamese Sentiment Analysis Data Utilities
# # data_utils.py
# import os
# import pickle
# import json
# import numpy as np
# import torch
# from torch.utils.data import Dataset, DataLoader
# from builders.dataset_builder import META_DATASET

# @META_DATASET.register()
# class VietnameseSentimentDataset(Dataset):
#     """
#     Dataset class for Vietnamese Sentiment Analysis
#     """
#     def __init__(self, data_file, tokenizer, sentiment_map=None):
#         with open(data_file, 'r', encoding='utf-8') as f:
#             data = json.load(f)
        
#         # Default sentiment mapping
#         if sentiment_map is None:
#             sentiment_map = {
#                 "Tích cực": 2,  # Positive: 2
#                 "Tiêu cực": 0,  # Negative: 0
#                 "Bình thường": 1  # Neutral: 1
#             }
            
#         category_map = {
#             "Chất lượng sản phẩm": 0,
#             "Dịch vụ": 1,
#             "Giá cả": 2,
#             "Khác": 3,
#         }
#         all_data = []
        
#         for item in data:
#             text = item['text']
            
#             for label_info in item['labels']:
#                 start_idx, end_idx, aspect_text, label = label_info
                
#                 # Extract category and sentiment
#                 category, sentiment = label.split('#')
#                 sentiment = sentiment.strip()
#                 category = category.strip()
                
#                 # Get the polarity from sentiment map
#                 polarity = sentiment_map.get(sentiment, 1)  # Default to neutral if not found
#                 category_idx = category_map.get(category, 3) # Default khác
                
#                 # Context is the entire text
#                 context = text
                
#                 # Find the context before and after the aspect
#                 left_context = text[:start_idx].strip()
#                 right_context = text[end_idx:].strip()
                
#                 # Create indices for LSTM models
#                 text_indices = tokenizer.text_to_sequence(text)
#                 context_indices = tokenizer.text_to_sequence(context)
#                 left_indices = tokenizer.text_to_sequence(left_context)
#                 right_indices = tokenizer.text_to_sequence(right_context, reverse=True)
#                 aspect_indices = tokenizer.text_to_sequence(aspect_text)
                
#                 # Create indices with aspects
#                 left_with_aspect_indices = tokenizer.text_to_sequence(left_context + " " + aspect_text)
#                 right_with_aspect_indices = tokenizer.text_to_sequence(aspect_text + " " + right_context, reverse=True)
                
#                 # Calculate aspect boundary
#                 left_len = np.sum(left_indices != 0)
#                 aspect_len = np.sum(aspect_indices != 0)
#                 aspect_boundary = np.asarray([left_len, left_len + aspect_len - 1], dtype=np.int64)
                
#                 data_item = {
#                     'text': text,
#                     'aspect': aspect_text,
#                     'category_id': category_idx,
#                     'category': category,
#                     'text_indices': text_indices,
#                     'context_indices': context_indices,
#                     'left_indices': left_indices,
#                     'left_with_aspect_indices': left_with_aspect_indices,
#                     'right_indices': right_indices,
#                     'right_with_aspect_indices': right_with_aspect_indices,
#                     'aspect_indices': aspect_indices,
#                     'aspect_boundary': aspect_boundary,
#                     'polarity': polarity,
#                     'span': [start_idx, end_idx, aspect_text, label]
#                 }
                
#                 all_data.append(data_item)
        
#         self.data = all_data

#     def _convert_to_indices(self, tensor_or_list):
#         """Convert tensor to numpy array or return as is"""
#         if torch.is_tensor(tensor_or_list):
#             return tensor_or_list.numpy()
#         return np.array(tensor_or_list)
    
#     def __getitem__(self, index):
#         return self.data[index]

#     def __len__(self):
#         return len(self.data)

# def load_data(data_file, tokenizer, sentiment_map=None, batch_size=16, shuffle=True):
#     """
#     Helper function to load data and create data loader
#     """
#     dataset = VietnameseSentimentDataset(
#         data_file=data_file,
#         tokenizer=tokenizer,
#         sentiment_map=sentiment_map
#     )
    
#     return DataLoader(
#         dataset=dataset,
#         batch_size=batch_size,
#         shuffle=shuffle
#     )

