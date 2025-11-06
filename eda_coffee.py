"""eda_coffee.py
Lightweight exploratory data analysis for data/Coffe_sales.csv.
Generates summary tables and plots (saved to outputs/).
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


DATA_PATH = Path("data") / "Coffe_sales.csv"
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    return df


def basic_summary(df: pd.DataFrame):
    print("Shape:", df.shape)
    print("\nDTypes:\n", df.dtypes)
    print("\nMissing values:\n", df.isna().sum())
    print("\nUnique counts:\n", df.nunique())


def price_by_coffee(df: pd.DataFrame, top_n=15):
    grp = df.groupby('coffee_name')['money']
    summary = grp.agg(['count', 'mean', 'std', 'min', 'max'])
    summary = summary.sort_values('count', ascending=False)
    summary.to_csv(OUT_DIR / 'price_by_coffee.csv')
    print('\nSaved price summary -> outputs/price_by_coffee.csv')
    # show coffees with multiple distinct prices (possible sizes/add-ons)
    prices = df.groupby('coffee_name')['money'].unique().apply(lambda x: sorted(x))
    multi_price = prices[prices.apply(len) > 1]
    print('\nCoffees with multiple prices (sample):')
    print(multi_price.head(top_n).to_string())
    return summary


def sales_counts(df: pd.DataFrame):
    vc = df['coffee_name'].value_counts()
    vc.to_csv(OUT_DIR / 'coffee_counts.csv')
    print('\nSaved coffee counts -> outputs/coffee_counts.csv')
    return vc


def time_series_revenue(df: pd.DataFrame):
    # ensure Date is datetime
    df['Date_parsed'] = pd.to_datetime(df['Date'], errors='coerce')
    df['revenue'] = df['money']
    daily = df.groupby('Date_parsed')['revenue'].sum().reset_index()
    daily.to_csv(OUT_DIR / 'daily_revenue.csv', index=False)
    print('\nSaved daily revenue -> outputs/daily_revenue.csv')
    # plot
    plt.figure(figsize=(10,4))
    sns.lineplot(data=daily, x='Date_parsed', y='revenue')
    plt.title('Daily revenue')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'daily_revenue.png')
    plt.close()
    print('Saved plot -> outputs/daily_revenue.png')
    return daily


def sales_by_hour(df: pd.DataFrame):
    hourly = df.groupby('hour_of_day').size().rename('count').reset_index()
    hourly.to_csv(OUT_DIR / 'hourly_counts.csv', index=False)
    plt.figure(figsize=(8,4))
    sns.barplot(data=hourly, x='hour_of_day', y='count', color='C0')
    plt.title('Sales count by hour_of_day')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'sales_by_hour.png')
    plt.close()
    print('\nSaved hourly sales -> outputs/sales_by_hour.png')
    return hourly


def price_distribution(df: pd.DataFrame):
    plt.figure(figsize=(8,4))
    sns.histplot(df['money'].dropna(), bins=30)
    plt.title('Money distribution')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'price_distribution.png')
    plt.close()
    print('Saved plot -> outputs/price_distribution.png')


def popular_coffees_boxplot(df: pd.DataFrame, top_n=8):
    top = df['coffee_name'].value_counts().index[:top_n]
    sub = df[df['coffee_name'].isin(top)]
    plt.figure(figsize=(10,5))
    sns.boxplot(data=sub, x='coffee_name', y='money')
    plt.xticks(rotation=45)
    plt.title(f'Price distribution for top {top_n} coffees')
    plt.tight_layout()
    plt.savefig(OUT_DIR / f'boxplot_top_{top_n}.png')
    plt.close()
    print(f'Saved boxplot -> outputs/boxplot_top_{top_n}.png')


def correlations(df: pd.DataFrame):
    # select numeric cols that may be informative
    numeric = df[['hour_of_day', 'money', 'Weekdaysort', 'Monthsort']].copy()
    corr = numeric.corr()
    corr.to_csv(OUT_DIR / 'numeric_correlations.csv')
    print('\nSaved numeric correlations -> outputs/numeric_correlations.csv')
    return corr


def run_all(path=DATA_PATH):
    df = load_data(path)
    basic_summary(df)
    sales_counts(df)
    price_by_coffee(df)
    time_series_revenue(df)
    sales_by_hour(df)
    price_distribution(df)
    popular_coffees_boxplot(df)
    corr = correlations(df)
    print('\nTop 10 coffees:\n', df['coffee_name'].value_counts().head(10))
    print('\nCorrelation matrix:\n', corr)


if __name__ == '__main__':
    print('Running EDA for', DATA_PATH)
    if not DATA_PATH.exists():
        raise SystemExit(f"Data file not found: {DATA_PATH} - check path")
    run_all()
