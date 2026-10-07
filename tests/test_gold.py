from gold import GoldAggregation

COLUMNS = [
    "product_name", "destination_city", "transaction_date",
    "demand_quantity", "ordered_quantity", "product_cost_usd",
    "transportation_cost_usd", "total_cost_usd", "available_inventory",
    "unit_price_usd", "product_category", "quantity_unit",
]


def test_two_supplier_rows_become_one_daily_row(spark):
    gold = GoldAggregation(spark=None, silver_table="unused", gold_table="unused")
    rows = [
        ("Gasoline", "Pune", "2026-01-01", 10.0, 12.0, 100.0, 10.0, 110.0, 500.0, 4.0, "Fuel", "L"),
        ("Gasoline", "Pune", "2026-01-01", 30.0, 33.0, 300.0, 30.0, 330.0, 800.0, 6.0, "Fuel", "L"),
    ]

    result = gold.aggregate(spark.createDataFrame(rows, COLUMNS)).collect()

    assert len(result) == 1                          # 2 rows -> 1 row
    row = result[0]
    assert row["demand_quantity"] == 40.0            # demand is added up
    assert row["available_inventory"] == 800.0      # stock level: biggest value, not the sum
    assert row["avg_unit_price_usd"] == 5.0         # price: average
    assert row["num_transactions"] == 2