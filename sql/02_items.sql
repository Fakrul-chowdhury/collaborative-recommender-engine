-- One row per product with its canonical description.
--
-- The same stock code appears with several spellings of its description, so the most
-- frequent one is kept. Descriptions are static item metadata used by the content model.
-- The purchase statistics here span the WHOLE period and are for EDA only: models must
-- derive popularity from the training window (see 04_interactions_window.sql).

DROP TABLE IF EXISTS items;

CREATE TABLE items AS
WITH description_counts AS (
    SELECT
        stock_code,
        description,
        ROW_NUMBER() OVER (
            PARTITION BY stock_code
            ORDER BY COUNT(*) DESC, description
        ) AS rank_in_item
    FROM clean_transactions
    WHERE description IS NOT NULL AND description <> ''
    GROUP BY stock_code, description
),
item_stats AS (
    SELECT
        stock_code,
        COUNT(DISTINCT customer_id) AS n_customers,
        COUNT(DISTINCT invoice)     AS n_invoices,
        MIN(invoice_date)           AS first_sold,
        MAX(invoice_date)           AS last_sold
    FROM clean_transactions
    GROUP BY stock_code
)
SELECT
    s.stock_code,
    d.description,
    s.n_customers,
    s.n_invoices,
    s.first_sold,
    s.last_sold
FROM item_stats AS s
LEFT JOIN description_counts AS d
    ON d.stock_code = s.stock_code AND d.rank_in_item = 1;

CREATE UNIQUE INDEX idx_items_code ON items (stock_code);
