import pandas as pd

def calculate_credit_to_debit_ratio(data, output_csv_path=None, show_result=True):

    # Calculate total credits (Amount > 0) and debits (Amount < 0) per account
    credits = data[data['Amount'] > 0].groupby('Account No')['Amount'].sum()
    debits = data[data['Amount'] < 0].groupby('Account No')['Amount'].sum().abs()

    # Combine into a single DataFrame
    credit_debit_df = pd.DataFrame({'Credit': credits, 'Debit': debits}).fillna(0)

    # Calculate Credit to Debit Ratio (add small constant to avoid division by zero)
    credit_debit_df['Credit_to_Debit_Ratio'] = credit_debit_df['Credit'] / (credit_debit_df['Debit'] + 1e-6)

    # Round all values to 2 decimal places
    credit_debit_df = credit_debit_df.round(4)

    # # Convert 'Account No' to int
    # credit_debit_df['Account No'] = credit_debit_df['Account No'].astype(int)

    # Print result if required
    if show_result:
        print(credit_debit_df)

    # Save to CSV if path provided
    if output_csv_path:
        credit_debit_df.to_csv(output_csv_path)

    return credit_debit_df

# Read the CSV file
data = pd.read_csv('cleaned_dataset2.csv')
result_df = calculate_credit_to_debit_ratio(
    data= data,
    output_csv_path="credit_debit_ratios.csv",
    show_result=True
)