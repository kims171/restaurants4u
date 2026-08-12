from pathlib import Path

import pandas as pd

OUTPUT_PATH = Path("data/production_logs/latest.parquet")


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    production_data = pd.DataFrame(
        {
            "Title": [
                "Production Restaurant A",
                "Production Restaurant B",
                "Production Restaurant C",
                "Production Restaurant D",
                "Production Restaurant E",
            ],
            "Category": [
                "Cafe",
                "Cafe",
                "Dessert",
                "Restaurant",
                "Fast Food",
            ],
            "Rating": [4.8, 4.6, 4.9, 3.7, 4.2],
            "Latitude": [43.6540, 43.6550, 43.6700, 43.6800, 43.6450],
            "Longitude": [-79.3810, -79.3820, -79.4000, -79.4100, -79.3700],
            "Website": [
                "https://prod-a.com",
                "https://prod-b.com",
                "https://prod-c.com",
                None,
                "https://prod-e.com",
            ],
            "Phone": [
                "1234567893",
                "1234567894",
                None,
                "1234567896",
                "1234567897",
            ],
        }
    )

    production_data.to_parquet(OUTPUT_PATH, index=False)
    print(f"Sample production logs written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
