import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("fake_transactional_dataset_1.csv")
print(df.isnull().sum())

# Convert data types
df['monopoly_money_amount'] = pd.to_numeric(df['monopoly_money_amount'], errors='coerce')
df['not_happened_yet_date'] = pd.to_datetime(df['not_happened_yet_date'], errors='coerce', format='%d/%m/%Y')
df['from_totally_fake_account'] = df['from_totally_fake_account'].astype(int)


# Handle missing values (drop because only 2 missing values)
df.dropna(subset=['to_randomly_generated_account'], inplace=True)
df.dropna(subset=['not_happened_yet_date'], inplace=True)



# Split transactions into user and merchant transactions
df_accounts = df[df['to_randomly_generated_account'].astype(str).str.isnumeric()]
df_merchants = df[~df['to_randomly_generated_account'].astype(str).str.isnumeric()]

# Identify whether the recipient is a user or a merchant and split into separate columns
df['Third Party Account No'] = df['to_randomly_generated_account'].astype(str).where(df['to_randomly_generated_account'].astype(str).str.isnumeric(), None)
df['Third Party Name'] = df['to_randomly_generated_account'].astype(str).where(~df['to_randomly_generated_account'].astype(str).str.isnumeric(), None)
df['Third Party Account No'].fillna('MTrx', inplace=True)
df['Third Party Name'].fillna('P2P', inplace=True)
df.drop(columns=['to_randomly_generated_account'], inplace=True)

# Sort transactions by sender account and time
df.sort_values(by=['from_totally_fake_account', 'not_happened_yet_date'], ascending=[True, True], inplace=True)


# Rename some columns
df.rename(columns={
    'from_totally_fake_account': 'Account No',
    'monopoly_money_amount': 'Amount',
    'not_happened_yet_date': 'Transaction_date'
}, inplace=True)

# Save the cleaned data
df.to_csv("cleaned_dataset1.csv", index=False)
