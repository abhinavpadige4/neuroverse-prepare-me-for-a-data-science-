import sqlite3
import pandas as pd

# ============================================================
# SQL Window Functions — Flipkart Data Science Interview Prep
# ============================================================
# Covers: ROW_NUMBER, RANK, DENSE_RANK, LAG, LEAD,
#         SUM/AVG OVER, NTILE, cumulative aggregates
# ============================================================

def create_sample_db():
    """Create in-memory SQLite DB with Flipkart-style e-commerce tables."""
    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE orders (
        order_id INTEGER PRIMARY KEY,
        user_id INTEGER,
        product_id INTEGER,
        category TEXT,
        order_date TEXT,
        amount REAL
    )
    """)

    cur.execute("""
    CREATE TABLE products (
        product_id INTEGER PRIMARY KEY,
        product_name TEXT,
        category TEXT,
        price REAL
    )
    """)

    orders_data = [
        (1, 101, 1, 'Electronics', '2024-01-05', 2999.0),
        (2, 101, 2, 'Electronics', '2024-01-12', 1499.0),
        (3, 101, 3, 'Clothing',    '2024-02-01', 499.0),
        (4, 102, 1, 'Electronics', '2024-01-03', 2999.0),
        (5, 102, 4, 'Home',        '2024-01-20', 899.0),
        (6, 102, 5, 'Home',        '2024-02-10', 1299.0),
        (7, 103, 2, 'Electronics', '2024-01-08', 1499.0),
        (8, 103, 3, 'Clothing',    '2024-01-15', 499.0),
        (9, 103, 6, 'Books',       '2024-02-05', 299.0),
        (10, 104, 1, 'Electronics','2024-01-02', 2999.0),
        (11, 104, 7, 'Clothing',   '2024-01-25', 799.0),
        (12, 104, 8, 'Books',      '2024-02-14', 199.0),
        (13, 105, 4, 'Home',       '2024-01-07', 899.0),
        (14, 105, 5, 'Home',       '2024-01-18', 1299.0),
        (15, 105, 6, 'Books',      '2024-02-20', 299.0),
    ]
    cur.executemany("INSERT INTO orders VALUES (?,?,?,?,?,?)", orders_data)

    products_data = [
        (1, 'Smartphone X', 'Electronics', 2999.0),
        (2, 'Headphones Pro','Electronics', 1499.0),
        (3, 'Cotton T-Shirt','Clothing',    499.0),
        (4, 'LED Lamp',      'Home',        899.0),
        (5, 'Coffee Maker',  'Home',        1299.0),
        (6, 'Python Guide',  'Books',       299.0),
        (7, 'Denim Jacket',  'Clothing',    799.0),
        (8, 'Data Science',  'Books',       199.0),
    ]
    cur.executemany("INSERT INTO products VALUES (?,?,?,?)", products_data)

    conn.commit()
    return conn


# ============================================================
# 1. ROW_NUMBER — Rank users by total spend
# ============================================================
def rank_users_by_spend(conn):
    """Assign row number to users ranked by total order amount."""
    query = """
    SELECT
        user_id,
        total_spend,
        ROW_NUMBER() OVER (ORDER BY total_spend DESC) AS spend_rank
    FROM (
        SELECT user_id, SUM(amount) AS total_spend
        FROM orders
        GROUP BY user_id
    )
    ORDER BY spend_rank
    """
    return pd.read_sql_query(query, conn)


# ============================================================
# 2. RANK & DENSE_RANK — Handle ties in category revenue
# ============================================================
def category_revenue_ranking(conn):
    """Rank categories by revenue using RANK and DENSE_RANK."""
    query = """
    SELECT
        category,
        total_revenue,
        RANK() OVER (ORDER BY total_revenue DESC) AS rank_val,
        DENSE_RANK() OVER (ORDER BY total_revenue DESC) AS dense_rank_val
    FROM (
        SELECT category, SUM(amount) AS total_revenue
        FROM orders
        GROUP BY category
    )
    ORDER BY rank_val
    """
    return pd.read_sql_query(query, conn)


# ============================================================
# 3. LAG & LEAD — Track order-to-order gaps per user
# ============================================================
def order_gaps_per_user(conn):
    """Calculate days between consecutive orders for each user."""
    query = """
    SELECT
        user_id,
        order_id,
        order_date,
        amount,
        LAG(order_date) OVER (PARTITION BY user_id ORDER BY order_date) AS prev_order_date,
        LEAD(order_date) OVER (PARTITION BY user_id ORDER BY order_date) AS next_order_date,
        julianday(order_date) - julianday(
            LAG(order_date) OVER (PARTITION BY user_id ORDER BY order_date)
        ) AS days_since_prev_order
    FROM orders
    ORDER BY user_id, order_date
    """
    return pd.read_sql_query(query, conn)


# ============================================================
# 4. Cumulative SUM — Running total per category
# ============================================================
def cumulative_category_revenue(conn):
    """Running total of revenue within each category over time."""
    query = """
    SELECT
        category,
        order_date,
        amount,
        SUM(amount) OVER (
            PARTITION BY category
            ORDER BY order_date
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS cumulative_revenue
    FROM orders
    ORDER BY category, order_date
    """
    return pd.read_sql_query(query, conn)


# ============================================================
# 5. Moving Average — 3-order rolling avg per user
# ============================================================
def moving_avg_per_user(conn):
    """3-order rolling average of amount per user."""
    query = """
    SELECT
        user_id,
        order_id,
        order_date,
        amount,
        AVG(amount) OVER (
            PARTITION BY user_id
            ORDER BY order_date
            ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING
        ) AS rolling_avg_3
    FROM orders
    ORDER BY user_id, order_date
    """
    return pd.read_sql_query(query, conn)


# ============================================================
# 6. NTILE — Split users into spend quartiles
# ============================================================
def user_spend_quartiles(conn):
    """Divide users into 4 spend quartiles using NTILE."""
    query = """
    SELECT
        user_id,
        total_spend,
        NTILE(4) OVER (ORDER BY total_spend) AS spend_quartile
    FROM (
        SELECT user_id, SUM(amount) AS total_spend
        FROM orders
        GROUP BY user_id
    )
    ORDER BY spend_quartile, total_spend
    """
    return pd.read_sql_query(query, conn)


# ============================================================
# 7. Top-N per group — Top 2 products per category by revenue
# ============================================================
def top_n_products_per_category(conn, n=2):
    """Get top N products per category ranked by total revenue."""
    query = f"""
    WITH product_revenue AS (
        SELECT
            category,
            product_id,
            SUM(amount) AS total_revenue,
            ROW_NUMBER() OVER (
                PARTITION BY category
                ORDER BY SUM(amount) DESC
            ) AS rn
        FROM orders
        GROUP BY category, product_id
    )
    SELECT category, product_id, total_revenue, rn
    FROM product_revenue
    WHERE rn <= {n}
    ORDER BY category, rn
    """
    return pd.read_sql_query(query, conn)


# ============================================================
# 8. First & Last order per user
# ============================================================
def first_last_order_per_user(conn):
    """Identify first and last order for each user using window functions."""
    query = """
    SELECT
        user_id,
        order_id,
        order_date,
        amount,
        FIRST_VALUE(order_date) OVER (
            PARTITION BY user_id ORDER BY order_date
        ) AS first_order_date,
        LAST_VALUE(order_date) OVER (
            PARTITION BY user_id
            ORDER BY order_date
            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
        ) AS last_order_date
    FROM orders
    ORDER BY user_id, order_date
    """
    return pd.read_sql_query(query, conn)


# ============================================================
# TESTS
# ============================================================
def run_tests():
    conn = create_sample_db()
    passed = 0
    total = 0

    # Test 1: ROW_NUMBER
    total += 1
    df = rank_users_by_spend(conn)
    assert len(df) == 5, f"Expected 5 users, got {len(df)}"
    assert df.iloc[0]['spend_rank'] == 1
    assert df['spend_rank'].is_monotonic_increasing
    passed += 1
    print("PASS: rank_users_by_spend")

    # Test 2: RANK / DENSE_RANK
    total += 1
    df = category_revenue_ranking(conn)
    assert len(df) == 4, f"Expected 4 categories, got {len(df)}"
    assert df.iloc[0]['rank_val'] == 1
    passed += 1
    print("PASS: category_revenue_ranking")

    # Test 3: LAG / LEAD
    total += 1
    df = order_gaps_per_user(conn)
    assert len(df) == 15
    first_row = df[df['user_id'] == 101].iloc[0]
    assert pd.isna(first_row['prev_order_date'])
    passed += 1
    print("PASS: order_gaps_per_user")

    # Test 4: Cumulative SUM
    total += 1
    df = cumulative_category_revenue(conn)
    assert len(df) == 15
    electronics = df[df['category'] == 'Electronics']
    assert electronics.iloc[-1]['cumulative_revenue'] == electronics['amount'].sum()
    passed += 1
    print("PASS: cumulative_category_revenue")

    # Test 5: Moving Average
    total += 1
    df = moving_avg_per_user(conn)
    assert len(df) == 15
    passed += 1
    print("PASS: moving_avg_per_user")

    # Test 6: NTILE
    total += 1
    df = user_spend_quartiles(conn)
    assert len(df) == 5
    assert df['spend_quartile'].max() <= 4
    passed += 1
    print("PASS: user_spend_quartiles")

    # Test 7: Top-N per category
    total += 1
    df = top_n_products_per_category(conn, n=2)
    for cat in df['category'].unique():
        assert len(df[df['category'] == cat]) <= 2
    passed += 1
    print("PASS: top_n_products_per_category")

    # Test 8: First/Last order
    total += 1
    df = first_last_order_per_user(conn)
    user101 = df[df['user_id'] == 101]
    assert user101.iloc[0]['first_order_date'] == '2024-01-05'
    assert user101.iloc[-1]['last_order_date'] == '2024-02-01'
    passed += 1
    print("PASS: first_last_order_per_user")

    conn.close()
    print(f"\n{'='*40}")
    print(f"Results: {passed}/{total} tests passed")
    print(f"{'='*40}")
    return passed == total


# ============================================================
# MAIN — Run all demos and tests
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("SQL Window Functions — Flipkart Interview Prep")
    print("=" * 60)

    conn = create_sample_db()

    print("\n--- 1. Users Ranked by Total Spend (ROW_NUMBER) ---")
    print(rank_users_by_spend(conn).to_string(index=False))

    print("\n--- 2. Category Revenue Ranking (RANK / DENSE_RANK) ---")
    print(category_revenue_ranking(conn).to_string(index=False))

    print("\n--- 3. Order Gaps Per User (LAG / LEAD) ---")
    print(order_gaps_per_user(conn).to_string(index=False))

    print("\n--- 4. Cumulative Revenue Per Category ---")
    print(cumulative_category_revenue(conn).to_string(index=False))

    print("\n--- 5. 3-Order Rolling Average Per User ---")
    print(moving_avg_per_user(conn).to_string(index=False))

    print("\n--- 6. User Spend Quartiles (NTILE) ---")
    print(user_spend_quartiles(conn).to_string(index=False))

    print("\n--- 7. Top 2 Products Per Category ---")
    print(top_n_products_per_category(conn, n=2).to_string(index=False))

    print("\n--- 8. First & Last Order Per User ---")
    print(first_last_order_per_user(conn).to_string(index=False))

    conn.close()

    print("\n" + "=" * 60)
    print("RUNNING TESTS")
    print("=" * 60)
    all_passed = run_tests()
    print(f"\nAll tests passed: {all_passed}")
