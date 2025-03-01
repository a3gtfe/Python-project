#Data Cleaning And Processing
import pandas as pd
from google.colab import drive

# Step 1: Mount Google Drive (Only for Google Colab)
drive.mount('/content/drive')

# Step 2: Load the dataset (Update the path as per your file location)
file_path = "/content/drive/MyDrive/train.csv"
df = pd.read_csv(file_path)

# Step 3: Display basic info about the dataset
print("Dataset Info:")
df.info()
#step4

# Fill missing values with mean for numerical columns
numerical_columns = df.select_dtypes(include=["number"]).columns
df[numerical_columns] = df[numerical_columns].apply(lambda x: x.fillna(x.mean()))

# Fill missing values with "Unknown" for text columns
text_columns = ["artists", "album_name", "track_name", "track_genre"]
df[text_columns] = df[text_columns].fillna("Unknown")
# Step 5: Convert text columns to lowercase for better analysis
text_columns = ["artists", "album_name", "track_name", "track_genre"]
df[text_columns] = df[text_columns].apply(lambda x: x.astype(str).str.lower())
# Fill missing values with mean for numerical columns
numerical_columns = df.select_dtypes(include=["number"]).columns
df[numerical_columns] = df[numerical_columns].apply(lambda x: x.fillna(x.mean()))

# Step 6: Save the cleaned file
cleaned_file_path = "/content/drive/MyDrive/trainee_cleaned.csv"
df.to_csv(cleaned_file_path, index=False)

print(f"✅ Cleaned dataset saved at: {cleaned_file_path}")
