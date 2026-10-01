-- Clean the raw invoice lines into purchase events usable as implicit feedback.
--
-- Dropped:
--   * cancelled invoices            (invoice number starts with 'C')
--   * returns / adjustments         (quantity <= 0)
--   * free or mispriced lines       (price <= 0)
--   * anonymous checkouts           (customer_id IS NULL) - no user to recommend to
--   * non-product stock codes       (POST, DOT, M, D, BANK CHARGES, gift vouchers, ...)
--
-- A product stock code is five digits optionally followed by letters (85123A, 15056BL).
-- SELECT DISTINCT removes exact duplicate lines; the two sheets of the source workbook
-- overlap for 1-9 December 2010, so those rows would otherwise be counted twice.

DROP TABLE IF EXISTS clean_transactions;

CREATE TABLE clean_transactions AS
SELECT DISTINCT
    TRIM(invoice)                 AS invoice,
    UPPER(TRIM(stock_code))       AS stock_code,
    TRIM(description)             AS description,
    CAST(quantity AS INTEGER)     AS quantity,
    invoice_date                  AS invoice_date,
    CAST(price AS REAL)           AS price,
    CAST(customer_id AS INTEGER)  AS customer_id,
    country                       AS country
FROM raw_transactions
WHERE customer_id IS NOT NULL
  AND UPPER(TRIM(invoice)) NOT LIKE 'C%'
  AND quantity > 0
  AND price > 0
  AND SUBSTR(TRIM(stock_code), 1, 5) GLOB '[0-9][0-9][0-9][0-9][0-9]'
  AND NOT SUBSTR(UPPER(TRIM(stock_code)), 6) GLOB '*[^A-Z]*';

CREATE INDEX idx_clean_date ON clean_transactions (invoice_date);
CREATE INDEX idx_clean_customer_item ON clean_transactions (customer_id, stock_code);
