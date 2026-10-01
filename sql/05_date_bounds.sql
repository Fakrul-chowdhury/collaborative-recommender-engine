-- First and last purchase timestamps, used to place the chronological split.

SELECT
    MIN(invoice_date) AS min_date,
    MAX(invoice_date) AS max_date
FROM clean_transactions;
