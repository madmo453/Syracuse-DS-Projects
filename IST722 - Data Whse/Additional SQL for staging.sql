use ist722_hhkhan_od3_stage

--Stage Dim Products
  select row_number() OVER (ORDER BY (SELECT 100)) AS ProductID,*
  from(
  into [ist722_hhkhan_od3_stage].[fudgecorporation].[stgDimProducts]
  from(
  select product_id AS CompanyProductID, product_name AS ProductName, product_department AS Department, product_retail_price AS RetailPrice, product_wholesale_price AS WholesalePrice, product_is_active AS Active, 1 AS Company
  from [fudgemart_v3].[dbo].[fm_products]
  UNION 
  select plan_id AS CompanyProductID, plan_name AS ProductName, NULL AS Department, plan_price AS RetailPrice, NULL AS WholesalePrice, plan_current AS Active, 2 AS Company
  from [fudgeflix_v3].[dbo].[ff_plans])a


  --Stage DimOrders
  select *
  into [ist722_hhkhan_od3_stage].[fudgecorporation].[stgDimOrders]
  from(
  select order_id AS OrderID, Customer_id AS CustomerID, order_date AS OrderDate, shipped_date AS ShippedDate, 1 AS Company
  from [fudgemart_v3].[dbo].[fm_orders]
  UNION 
  select ab_id AS OrderID, ab_account_id AS CustomerID, ab_date AS OrderDate, NULL AS ShippedDate, 2 AS Company
  from [fudgeflix_v3].[dbo].[ff_account_billing])a

  --Stage DimOrderDetails
  select *
  into [ist722_hhkhan_od3_stage].[fudgecorporation].[stgDimOrderDetails]
  from(
  select order_id AS OrderID, product_ID AS ProductID, order_qty AS OrderQuanity, 1 AS Company
  from  [fudgemart_v3].[dbo].[fm_order_details]
  UNION 
  select ab_id AS OrderID, ab_plan_id AS ProductID, 1 AS OrderQuanity, 2 AS Company
  from [fudgeflix_v3].[dbo].[ff_account_billing])a
  

  --Stage DimCompany
  CREATE TABLE [ist722_hhkhan_od3_stage].[fudgecorporation].[stgDimCompany] (CompanyName nvarchar(20) NOT NULL, CompanyAbbrev nvarchar(2) NOT NULL)
  INSERT INTO  [ist722_hhkhan_od3_stage].[fudgecorporation].[stgDimCompany] (CompanyName, CompanyAbbrev)
  VALUES ('FudgeMart', 'FM'),('FudgeFlix', 'FF')

  --Stage DimDate
  select *
  into [ist722_hhkhan_od3_stage].[fudgecorporation].[stgDimDate]
  from [ExternalSources2].[dbo].[date_dimension]
  WHERE Year between 2006 and 2013

  --Stage DimCustomers
  select *
  into [ist722_hhkhan_od3_stage].[fudgecorporation].[stgDimCustomers]
  from(
  select customer_id AS CustomerID, customer_email AS Email, NULL AS AccountProduct, 1 AS Company 
  from [fudgemart_v3].[dbo].[fm_customers]
  UNION 
  select account_id AS CustomerID, account_email AS Email,  account_plan_id AS AccountProduct, 2 AS Company
  from [fudgeflix_v3].[dbo].[ff_accounts])a

  --Stage FactSales
  select *
  into [ist722_hhkhan_od3_stage].[fudgecorporation].[stgFactSales]
  from (
  select c.product_ID AS ProductID, Customer_id AS CustomerID, a.order_id AS OrderID, order_date AS OrderDate, shipped_date AS ShippedDate, 1 AS Company, order_qty AS OrderQuantity, 
  product_retail_price AS RetailPrice
  from [fudgemart_v3].[dbo].[fm_orders]a JOIN  [fudgemart_v3].[dbo].[fm_order_details] b ON (a.order_id = b.order_Id) JOIN [fudgemart_v3].[dbo].[fm_products] c ON (b.product_id = c.product_id)
  UNION
  select plan_id AS ProductID, account_id AS CustomerID, ab_id AS OrderID, ab_date AS OrderDate, NULL AS ShippedDate, 2 AS Company, 1 AS OrderQuanity, plan_price AS RetailPrice
  from [fudgeflix_v3].[dbo].[ff_account_billing] a JOIN [fudgeflix_v3].[dbo].[ff_plans] b ON (a.ab_plan_id = b.plan_id) JOIN [fudgeflix_v3].[dbo].[ff_accounts] c ON (a.ab_account_id = c.account_id))a