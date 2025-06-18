from typing import List
from utils.instance import Instance, InstanceList

def collate_fn(items: List[Instance]) -> InstanceList:
    batch = InstanceList(items)
    
    # Add spans and original_text if available in items
    if hasattr(items[0], 'spans'):
        batch.spans = [item.spans for item in items]
    else:
        # If no spans, consider entire text as 1 span
        batch.spans = []
        for item in items:
            if hasattr(item, 'original_text'):
                text_len = len(item.original_text)
                batch.spans.append([(0, text_len)])
            else:
                # Fallback: use a default span
                batch.spans.append([(0, 100)])  # Adjust as needed
    
    if hasattr(items[0], 'original_text'):
        batch.original_text = [item.original_text for item in items]
    elif hasattr(items[0], 'text'):
        batch.original_text = [item.text for item in items]
    else:
        # Fallback: create placeholder text
        batch.original_text = [f"text_{i}" for i in range(len(items))]
    
    return batch