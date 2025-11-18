import os
import pandas as pd

# Set up the root directory path
root_dir = "C:\\Users\\habib\\Downloads\\dataset\\Skin Disease"  # replace with the actual path to the dataset

# Initialize an empty list to store data
data = []

# Iterate through train, test, and validation splits
for split in ['train', 'test', 'validation']:
    split_dir = os.path.join(root_dir, split)
    
    # Iterate through tinea and melanoma categories
    for category in ['Tinea', 'Melanoma']:
        category_dir = os.path.join(split_dir, category)
        
        # Get all image filenames in the category folder
        for filename in os.listdir(category_dir):
            if filename.endswith('.jpg') or filename.endswith('.png'):  # adjust extensions if needed
                imagepath = os.path.join(category_dir, filename)
                label = 1 if category == 'Tinea' else 0  # label 1 for tinea, 0 for melanoma
                
                # Append the data to the list
                data.append({
                    'imagepath': imagepath,
                    'filename': filename,
                    'disease_name': 'skin disease',
                    'category': category,
                    'split': split,
                    'label': label
                })

# Convert the list to a pandas DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv('skin_disease_dataset.csv', index=False)

print("CSV file has been created successfully!")
