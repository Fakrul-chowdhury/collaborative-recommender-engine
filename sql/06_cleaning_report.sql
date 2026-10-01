-- How many raw rows each cleaning rule removes (rules overlap, so counts do not add up).

SELECT
    COUNT(*)                                                        AS raw_rows,
    SUM(customer_id IS NULL)                                        AS null_customer,
    SUM(UPPER(TRIM(invoice)) LIKE 'C%')                             AS cancelled_invoice,
    SUM(quantity <= 0)                                              AS non_positive_quantity,
    SUM(price <= 0)                                                 AS non_positive_price,
    SUM(NOT (SUBSTR(TRIM(stock_code), 1, 5) GLOB '[0-9][0-9][0-9][0-9][0-9]'
             AND NOT SUBSTR(UPPER(TRIM(stock_code)), 6) GLOB '*[^A-Z]*'))
                                                                    AS non_product_code,
    (SELECT COUNT(*) FROM clean_transactions)                       AS clean_rows,
    (SELECT COUNT(DISTINCT customer_id) FROM clean_transactions)    AS customers,
    (SELECT COUNT(DISTINCT stock_code) FROM clean_transactions)     AS products,
    (SELECT COUNT(DISTINCT invoice) FROM clean_transactions)        AS invoices,
    (SELECT COUNT(*) FROM interactions)                             AS interactions
FROM raw_transactions;
