-- User-item interactions restricted to the half-open time window [:start, :end).
--
-- Every train / validation / test set is produced by this one query with different
-- bounds, which is what guarantees the split is chronological and leak-free.

SELECT
    customer_id,
    stock_code,
    COUNT(DISTINCT invoice) AS purchase_count,
    SUM(quantity)           AS total_quantity,
    MIN(invoice_date)       AS first_purchase,
    MAX(invoice_date)       AS last_purchase
FROM clean_transactions
WHERE invoice_date >= :start
  AND invoice_date <  :end
GROUP BY customer_id, stock_code
ORDER BY customer_id, stock_code;
