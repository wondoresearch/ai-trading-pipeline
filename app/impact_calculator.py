import pandas as pd


def calculate_return(
    entry_price,
    exit_price
):

    if entry_price is None:
        return None

    return (
        (exit_price - entry_price)
        / entry_price
    )


def calculate_impact(
    event,
    prices
):

    if prices.empty:
        return None

    published = pd.to_datetime(
        event["published_at"],
        utc=True,
        errors="coerce"
    )

    if pd.isna(published):
        return None

    published = (
        published
        .tz_convert("Asia/Jakarta")
        .tz_localize(None)
    )

    news_date = published.normalize()

    prices = prices.copy()

    prices["Date"] = pd.to_datetime(
        prices["Date"]
    ).dt.normalize()

    # Ambil trading day pertama setelah berita
    future_prices = prices[
        prices["Date"] > news_date
    ].sort_values("Date")

    if future_prices.empty:
        return None

    # Entry = close trading day pertama setelah news
    entry_row = future_prices.iloc[0]

    entry_price = float(
        entry_row["Close"]
    )

    entry_date = entry_row["Date"]

    result = {
        "news_id":
            event["news_id"],

        "ticker":
            event["ticker"],

        "title":
            event["title"],

        "published_at":
            event["published_at"],

        "market_entry_date":
            entry_date.strftime(
                "%Y-%m-%d"
            ),

        "entry_price":
            entry_price,

        "sentiment":
            event["sentiment"],

        "sentiment_score":
            event["sentiment_score"],

        "signed_score":
            event["signed_score"]
    }

    horizons = {
        "return_1d": 1,
        "return_3d": 3,
        "return_5d": 5,
        "return_10d": 10
    }

    for field, days in horizons.items():

        if len(future_prices) > days:

            exit_price = float(
                future_prices.iloc[
                    days
                ]["Close"]
            )

            result[field] = calculate_return(
                entry_price,
                exit_price
            )

        else:

            result[field] = None

    return result