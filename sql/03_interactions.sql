-- User-item interaction table over the whole period.
--
-- purchase_count is the number of distinct invoices on which the customer bought the
-- item (purchase occasions). It is a better implicit signal than summed quantity, which
-- in this wholesale-heavy data mostly reflects pack sizes.
--
-- This full-period table is for EDA and reporting. Training data is built per time
-- window by 04_interactions_window.sql so that nothing after a split boundary leaks in.

DROP TABLE IF EXISTS interactions;

CREATE TABLE interactions AS
SELECT
    customer_id,
    stock_code,
    COUNT(DISTINCT invoice) AS purchase_count,
    SUM(quantity)           AS total_quantity,
    MIN(invoice_date)       AS first_purchase,
    MAX(invoice_date)       AS last_purchase
FROM clean_transactions
GROUP BY customer_id, stock_code;

CREATE INDEX idx_interactions_customer ON interactions (customer_id);
CREATE INDEX idx_interactions_item ON interactions (stock_code);
