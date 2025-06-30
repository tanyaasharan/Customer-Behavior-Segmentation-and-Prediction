import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
import seaborn as sns
from scipy.stats import gaussian_kde
import matplotlib.cm as cm

def load_and_preprocess_data(file_path):
    # Load the dataset and parse the transaction date
    df = pd.read_csv(file_path)
    df['Transaction_date'] = pd.to_datetime(df['Transaction_date'], errors='coerce')
    return df

def plot_daily_spending(df):
    # Filter January transactions and sort by date
    date_format = mdates.DateFormatter('%b %d')
    df_january = df[df['Transaction_date'].dt.month == 1].sort_values('Transaction_date')

    # Aggregate daily spending
    daily_spending_jan = df_january.groupby('Transaction_date', as_index=False)['Amount'].sum()

    # Plot line chart of total daily spending in January
    plt.figure(figsize=(10, 5))
    plt.plot(daily_spending_jan['Transaction_date'], daily_spending_jan['Amount'], marker='o')
    plt.xlabel('Date')
    plt.ylabel('Total Spending')
    plt.title('Total Daily Spending in January')
    plt.gca().xaxis.set_major_formatter(date_format)
    plt.xticks(rotation=45)
    plt.grid()
    plt.tight_layout()
    plt.show()

    # Plot line chart of total daily spending
    df_sorted = df.sort_values('Transaction_date')
    daily_spending_all = df_sorted.groupby('Transaction_date', as_index=False)['Amount'].sum()
    plt.figure(figsize=(12, 5))
    plt.plot(daily_spending_all['Transaction_date'], daily_spending_all['Amount'], marker='o', linestyle='-')
    plt.xlabel('Date')
    plt.ylabel('Total Spending')
    plt.title('Total Daily Spending (All Data)')
    plt.gca().xaxis.set_major_formatter(date_format)
    plt.xticks(rotation=45)
    plt.grid()
    plt.tight_layout()
    plt.show()

def plot_amount_histogram(df):
    # Plot histogram of transaction amounts (limited x-axis for clarity)
    amounts = df["Amount"]
    amounts = amounts[(amounts > 0) & (amounts < 60)]
    plt.figure(figsize=(10, 6))
    plt.hist(amounts, bins=100, edgecolor='black', color='skyblue', alpha=0.6, density=True)
    kde = gaussian_kde(amounts, bw_method=0.1)
    x_vals = np.linspace(0, 60, 500)
    plt.plot(x_vals, kde(x_vals), color='blue', linewidth=2)
    plt.xlabel("Transaction Amount")
    plt.ylabel("Density")
    plt.title("Distribution of Most Transaction Amounts (with KDE)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def classify_categories(df):
    # Clear specific irrelevant or special-case values
    df.loc[df["Third Party Account No"] == "MTrx", "Third Party Account No"] = np.nan
    df.loc[df["Third Party Name"] == "P2P", "Third Party Name"] = np.nan

    # Define transaction category mapping
    categories = {
        "Bars & Beverage": ["COCKTAIL_BAR", "BAR", "PUB", "LOCAL_WATERING_HOLE", "WINE_BAR", "WHISKEY_BAR", "G&T_BAR", "WHISKEY_SHOP"],
        "Coffee & Tea": ["A_LOCAL_COFFEE_SHOP", "COFFEE_SHOP", "GOURMET_COFFEE_SHOP", "A_CAFE", "PRETENTIOUS_COFFEE_SHOP", "HIPSTER_COFFEE_SHOP", "TOTALLY_A_REAL_COFFEE_SHOP", "TO_BEAN_OR_NOT_TO_BEAN", "TEA_SHOP"],
        "Food & Dining": ["SEAFOOD_RESTAURANT", "INDIAN_RESTAURANT", "RESTAURANT", "LOCAL_RESTAURANT", "STEAK_HOUSE", "CHINESE_RESTAURANT", "CHINESE_TAKEAWAY", "TAKEAWAY", "TAKEAWAY_CURRY", "LUNCH_VAN", "LUNCH_PLACE", "SANDWICH_SHOP"],
        "Retail Stores": ["CLOTHES_SHOP", "FASHION_SHOP", "KIDS_CLOTHING_SHOP", "KIDS_ACTIVITY_CENTRE", "JEWLLERY_SHOP", "ACCESSORY_SHOP", "FASHIONABLE_SPORTSWARE_SHOP", "PET_SHOP", "PET_TOY_SHOP", "TOY_SHOP", "GREENGROCER", "BUTCHER", "BUTCHERS", "COOKSHOP", "LIQUOR_STORE", "FLORIST"],
        "Electronics & Books": ["ELECTRONICS_SHOP", "TECH_SHOP", "HIPSTER_ELECTRONICS_SHOP", "VIDEO_GAME_STORE", "GAME_SHOP", "COMIC_BOOK_SHOP", "NERDY_BOOK_STORE", "SECOND_HAND_BOOKSHOP", "BOOKSHOP", "LOCAL_BOOKSHOP", "SCHOOL_SUPPLY_STORE"],
        "Supermarkets": ["EXPRESS_SUPERMARKET", "THE_SUPERMARKET", "LARGE_SUPERMARKET", "A_SUPERMARKET", "GREENGROCER"],
        "Home Improvement & DIY": ["HOME_IMPROVEMENT_STORE", "DIY_STORE"],
        "Entertainment & Leisure": ["CINEMA", "STREAMING_SERVICE"],
        "Health & Fitness": ["GYM", "SPORT_SHOP", "RUNNING_SHOP", "FASHIONABLE_SPORTSWARE_SHOP"],
        "Kids & Family": ["CHILDRENDS_SHOP", "KIDS_CLOTHING_SHOP", "KIDS_ACTIVITY_CENTRE"],
        "Food Vouchers & Gifts": ["RESTAURANT_VOUCHER"],
        "Other": ["ROASTERIE", "TO_BEAN_OR_NOT_TO_BEAN"]
    }

    # Map merchant types to category group
    category_mapping = {item: cat for cat, items in categories.items() for item in items}
    df["Category Group"] = df["Third Party Name"].map(category_mapping).fillna("Uncategorized")

    # Override with 'To Account Transfers' if the destination account is not null
    df.loc[df["Third Party Account No"].notna(), "Category Group"] = "To Account Transfers"
    return df

def plot_transaction_amount_and_count(df):
    # Sum transaction amount per category group
    category_sums = df.groupby("Category Group")["Amount"].sum().sort_values(ascending=False)

    # Count transactions per category group
    category_counts = df["Category Group"].value_counts()

    # Bar plot: total transaction amount per category
    # plt.figure(figsize=(14, 8))
    # category_sums.sort_values(ascending=False).plot(kind='bar', color='skyblue', edgecolor='black')
    # plt.ylabel("Total Transaction Amount")
    # plt.xlabel("Category Group")
    # plt.title("Total Transaction Amount by Category")
    # plt.xticks(rotation=20, ha='right')
    # plt.grid(axis='y', linestyle='--', alpha=0.7)
    # plt.show()

    colors = cm.viridis(np.linspace(0, 1, len(category_sums)))
    plt.figure(figsize=(10, 8))
    category_sums.plot(kind='barh', color=colors, edgecolor='black')
    plt.xlabel("Total Transaction Amount")
    plt.ylabel("Category Group")
    plt.title("Total Transaction Amount by Category")
    plt.gca().invert_yaxis()  # 让最大值在上面
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


    # # Pie chart: distribution of transaction counts
    # plt.figure(figsize=(12, 8))
    # explode = [0.05 if x < 5 else 0 for x in category_counts]
    # wedges, _ = plt.pie(category_counts, labels=None, colors=plt.cm.Paired.colors, startangle=140,
    #                     wedgeprops={'edgecolor': 'black'}, explode=explode)
    # legend_labels = [f"{label} ({count / sum(category_counts) * 100:.1f}%)" for label, count in zip(category_counts.index, category_counts)]
    # plt.legend(wedges, legend_labels, loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10, title="Categories")
    # plt.title("Transaction Count Distribution")
    # plt.subplots_adjust(left=0.1, right=0.75)
    # plt.show()

def classify_user_activity(df):
    # Count number of transactions per user
    user_transaction_counts = df["account No"].value_counts()
    low_threshold = user_transaction_counts.quantile(1/4)
    high_threshold = user_transaction_counts.quantile(3/4)

    # Classify users based on activity frequency
    def categorize_user(txn_count):
        if txn_count <= low_threshold:
            return "Low Frequency Users"
        elif txn_count <= high_threshold:
            return "Medium Frequency Users"
        else:
            return "High Frequency Users"

    df["User Category"] = df["account No"].map(lambda x: categorize_user(user_transaction_counts[x]))
    return df

def plot_user_category_summaries(df):
    # Step 1: Count unique users in each user activity group
    user_counts = df.groupby("User Category")["account No"].nunique()

    # Step 2: Calculate total and average (per capita) amount per category/user type
    category_totals = df.groupby(["Category Group", "User Category"])["amount"].sum()
    user_category_summary = df.groupby(["User Category", "Category Group"])["amount"].sum().unstack(fill_value=0)
    user_category_counts = df.groupby(["User Category", "Category Group"]).size().unstack(fill_value=0)

    category_per_capita = category_totals / user_counts
    category_per_capita = category_per_capita.unstack(fill_value=0)

    # Bar chart: average transaction amount per user, per category
    fig, ax = plt.subplots(figsize=(14, 8))
    category_per_capita.plot(kind='bar', width=0.8, edgecolor='black', ax=ax)
    plt.ylabel("Average Transaction Amount per User")
    plt.xlabel("Category Group")
    plt.title("Per Capita Transaction Amount by Category and User Activity Level")
    plt.xticks(rotation=20, ha='right')
    plt.legend(title="User Activity Level")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    # Individual bar chart: total transaction amount per category per user group
    for user_category in user_category_summary.index:
        plt.figure(figsize=(14, 8))
        user_category_summary.loc[user_category].sort_values(ascending=False).plot(kind='bar', color='lightcoral', edgecolor='black')
        plt.ylabel("Total Transaction Amount")
        plt.xlabel("Category Group")
        plt.title(f"Total Transaction Amount for {user_category}")
        plt.xticks(rotation=20, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

    # Individual pie chart: transaction count distribution per user group
    for user_category in user_category_counts.index:
        category_counts = user_category_counts.loc[user_category]
        sorted_counts = category_counts.sort_values(ascending=False)
        custom_colors = plt.cm.tab20.colors[:len(sorted_counts)]
        explode = [0.05 if x < 5 else 0 for x in sorted_counts]

        plt.figure(figsize=(12, 8))
        wedges, _ = plt.pie(sorted_counts, labels=None, colors=custom_colors, startangle=140,
                            wedgeprops={'edgecolor': 'black'}, explode=explode)

        legend_labels = [f"{label} ({count / sum(sorted_counts) * 100:.1f}%)" for label, count in zip(sorted_counts.index, sorted_counts)]
        plt.legend(wedges, legend_labels, loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10, title="Categories")
        plt.title(f"Transaction Count Distribution for {user_category}")
        plt.subplots_adjust(left=0.1, right=0.75)
        plt.show()

if __name__ == "__main__":
    path = "cleaned_dataset1.csv"
    df = load_and_preprocess_data(path)
    df = classify_categories(df)
    plot_daily_spending(df)
    plot_amount_histogram(df)
    plot_transaction_amount_and_count(df)
