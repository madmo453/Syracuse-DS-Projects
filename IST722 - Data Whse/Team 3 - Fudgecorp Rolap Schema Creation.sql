

--ROLAP schema generation. Use to create tables to populate for ETL Load.
use ist722_hhkhan_od3_dw
go
DROP TABLE fudgecorporation.FactSales 
DROP TABLE fudgecorporation.FactEmployeeExpenses
DROP TABLE fudgecorporation.DimOrderDetails 
DROP TABLE fudgecorporation.DimDate 
DROP TABLE fudgecorporation.DimProducts 
DROP TABLE fudgecorporation.DimOrders 
DROP TABLE fudgecorporation.DimCustomers 
DROP TABLE fudgecorporation.DimCompany 
DROP TABLE fudgecorporation.DimEmployees

------------------------------------------------------------------------------ Dimension Table Creation --------------------------------------------------------------------------------------------------------------------------
/* Create table fudgecorporation.DimEmployees */
CREATE TABLE fudgecorporation.DimEmployees (
   [EmployeeKey]  int IDENTITY  NOT NULL
,  [EmployeeID]  int   NOT NULL
,  [Jobtitle]  varchar(20)   NOT NULL
,  [EmployeeDepartment]  varchar(20)   NOT NULL
,  [HourlyWage]  money   NOT NULL
,  [Fulltime]  bit   NOT NULL
,  [CompanyID]  int   NOT NULL
, CONSTRAINT [PK_fudgecorporation.DimEmployees] PRIMARY KEY CLUSTERED 
( [EmployeeKey] )
) ON [PRIMARY]
;

SET IDENTITY_INSERT fudgecorporation.DimEmployees ON
;
INSERT INTO fudgecorporation.DimEmployees (employeeKey, employeeID, Jobtitle, EmployeeDepartment, HourlyWage, Fulltime, CompanyID)
VALUES (-1, -1, '', '', '', '','')
;
SET IDENTITY_INSERT fudgecorporation.DimEmployees OFF
;


/* Drop table fudgecorporation.DimDate */
IF EXISTS (SELECT * FROM dbo.sysobjects WHERE id = OBJECT_ID(N'fudgecorporation.DimDate') AND OBJECTPROPERTY(id, N'IsUserTable') = 1)
DROP TABLE fudgecorporation.DimDate 
;

/* Create table fudgecorporation.DimDate */
CREATE TABLE fudgecorporation.DimDate (
   [DateKey]  int   NOT NULL
,  [Date]  datetime   NULL
,  [FullDateUSA]  nchar(11)   NOT NULL
,  [DayOfWeek]  tinyint   NOT NULL
,  [DayName]  nchar(10)   NOT NULL
,  [DayOfMonth]  tinyint   NOT NULL
,  [DayOfYear]  int   NOT NULL
,  [WeekOfYear]  tinyint   NOT NULL
,  [MonthName]  nchar(10)   NOT NULL
,  [MonthOfYear]  tinyint   NOT NULL
,  [Quarter]  tinyint   NOT NULL
,  [QuarterName]  nchar(10)   NOT NULL
,  [Year]  int   NOT NULL
,  [IsWeekday]  bit  DEFAULT 0 NOT NULL
, CONSTRAINT [PK_fudgecorporation.DimDate] PRIMARY KEY CLUSTERED 
( [DateKey] )
) ON [PRIMARY]
;

INSERT INTO fudgecorporation.DimDate (DateKey, Date, FullDateUSA, DayOfWeek, DayName, DayOfMonth, DayOfYear, WeekOfYear, MonthName, MonthOfYear, Quarter, QuarterName, Year, IsWeekday)
VALUES (-1, '', 'Unk date', 0, 'Unk date', 0, 0, 0, 'Unk month', 0, 0, 'Unk qtr', 0, 0)
;

/* Drop table fudgecorporation.DimProducts */
IF EXISTS (SELECT * FROM dbo.sysobjects WHERE id = OBJECT_ID(N'fudgecorporation.DimProducts') AND OBJECTPROPERTY(id, N'IsUserTable') = 1)
DROP TABLE fudgecorporation.DimProducts 
;

/* Create table fudgecorporation.DimProducts */
CREATE TABLE fudgecorporation.DimProducts (
   [ProductKey]  int IDENTITY  NOT NULL
,  [ProductID]  int   NOT NULL
,  [ProductName]  nvarchar(50)   NOT NULL
,  [Department]  nvarchar(50)   NULL
,  [RetailPrice]  money   NOT NULL
,  [WholesalePrice]  money   NULL
,  [Active]  bit   NOT NULL
,  [CompanyID]  int   NOT NULL
, CONSTRAINT [PK_fudgecorporation.DimProducts] PRIMARY KEY CLUSTERED 
( [ProductKey] )
) ON [PRIMARY]
;

SET IDENTITY_INSERT fudgecorporation.DimProducts ON
;
INSERT INTO fudgecorporation.DimProducts (ProductKey, ProductID, ProductName, Department, RetailPrice, WholesalePrice, Active, CompanyID)
VALUES (-1, -1, '', '', '', '', '', '')
;
SET IDENTITY_INSERT fudgecorporation.DimProducts OFF
;


/* Drop table fudgecorporation.DimOrders */
IF EXISTS (SELECT * FROM dbo.sysobjects WHERE id = OBJECT_ID(N'fudgecorporation.DimOrders') AND OBJECTPROPERTY(id, N'IsUserTable') = 1)
DROP TABLE fudgecorporation.DimOrders 
;

/* Create table fudgecorporation.DimOrders */
CREATE TABLE fudgecorporation.DimOrders (
   [OrderKey]  int IDENTITY  NOT NULL
,  [OrderID]  int   NOT NULL
,  [CustomerID]  int   NOT NULL
,  [OrderDate]  datetime   NOT NULL
,  [ShippedDate]  datetime   NULL
,  [CompanyID]  int   NOT NULL
, CONSTRAINT [PK_fudgecorporation.DimOrders] PRIMARY KEY CLUSTERED 
( [OrderKey] )
) ON [PRIMARY]
;

SET IDENTITY_INSERT fudgecorporation.DimOrders ON
;
INSERT INTO fudgecorporation.DimOrders (OrderKey, OrderID, CustomerID, OrderDate, ShippedDate, CompanyID)
VALUES (-1, -1, '', '', '', '')
;
SET IDENTITY_INSERT fudgecorporation.DimOrders OFF
;

/* Drop table fudgecorporation.DimCustomers */
IF EXISTS (SELECT * FROM dbo.sysobjects WHERE id = OBJECT_ID(N'fudgecorporation.DimCustomers') AND OBJECTPROPERTY(id, N'IsUserTable') = 1)
DROP TABLE fudgecorporation.DimCustomers 
;

/* Create table fudgecorporation.DimCustomers */
CREATE TABLE fudgecorporation.DimCustomers (
   [CustomerKey]  int IDENTITY  NOT NULL
,  [CustomerID]  int   NOT NULL
,  [Email]  nvarchar(200)   NOT NULL
,  [AccountProduct]  int NULL
,  [CompanyID]  int   NOT NULL
, CONSTRAINT [PK_fudgecorporation.DimCustomers] PRIMARY KEY CLUSTERED 
( [CustomerKey] )
) ON [PRIMARY]
;

SET IDENTITY_INSERT fudgecorporation.DimCustomers ON
;
INSERT INTO fudgecorporation.DimCustomers (CustomerKey, CustomerID, Email, AccountProduct, CompanyID)
VALUES (-1, -1, '', '', '')
;
SET IDENTITY_INSERT fudgecorporation.DimCustomers OFF
;

/* Drop table fudgecorporation.DimCompany */
IF EXISTS (SELECT * FROM dbo.sysobjects WHERE id = OBJECT_ID(N'fudgecorporation.DimCompany') AND OBJECTPROPERTY(id, N'IsUserTable') = 1)
DROP TABLE fudgecorporation.DimCompany 
;

/* Create table fudgecorporation.DimCompany */
CREATE TABLE fudgecorporation.DimCompany (
   [CompanyKey]  int IDENTITY  NOT NULL
,  [CompanyID]  int  NOT NULL
,  [CompanyName]  nvarchar(20)   NOT NULL
,  [CompanyAbbrev]  nvarchar(2)   NOT NULL
, CONSTRAINT [PK_fudgecorporation.DimCompany] PRIMARY KEY CLUSTERED 
( [CompanyKey] )
) ON [PRIMARY]
;

SET IDENTITY_INSERT fudgecorporation.DimCompany ON
;
INSERT INTO fudgecorporation.DimCompany (CompanyKey, CompanyID, CompanyName, CompanyAbbrev)
VALUES (-1, '', '', '')
;
SET IDENTITY_INSERT fudgecorporation.DimCompany OFF
;

/* Drop table fudgecorporation.DimOrderDetails */
IF EXISTS (SELECT * FROM dbo.sysobjects WHERE id = OBJECT_ID(N'fudgecorporation.DimOrderDetails') AND OBJECTPROPERTY(id, N'IsUserTable') = 1)
DROP TABLE fudgecorporation.DimOrderDetails 
;

/* Drop table fudgecorporation.DimOrderDetails */
IF EXISTS (SELECT * FROM dbo.sysobjects WHERE id = OBJECT_ID(N'fudgecorporation.DimOrderDetails') AND OBJECTPROPERTY(id, N'IsUserTable') = 1)
DROP TABLE fudgecorporation.DimOrderDetails 
;

/* Create table fudgecorporation.DimOrderDetails */
CREATE TABLE fudgecorporation.DimOrderDetails (
   [OrderID]  int   NOT NULL
,  [ProductID]  int   NOT NULL
,  [QrderQty]  int   NOT NULL
,  [CompanyID]  int   NOT NULL
, CONSTRAINT [PK_fudgecorporation.DimOrderDetails] PRIMARY KEY CLUSTERED 
( [OrderID], [ProductID], [CompanyID] )
) ON [PRIMARY]
;

ALTER TABLE fudgecorporation.DimOrderDetails ADD CONSTRAINT
   FK_fudgecorporation_DimOrderDetails_OrderID FOREIGN KEY
   (
   OrderID
   ) REFERENCES fudgecorporation.DimOrders
   ( OrderKey )
     ON UPDATE  NO ACTION
     ON DELETE  NO ACTION
;

ALTER TABLE fudgecorporation.DimOrderDetails ADD CONSTRAINT
   FK_fudgecorporation_DimOrderDetails_CompanyID FOREIGN KEY
   (
   CompanyID
   ) REFERENCES fudgecorporation.DimCompany
   ( CompanyKey )
     ON UPDATE  NO ACTION
     ON DELETE  NO ACTION
;
ALTER TABLE fudgecorporation.DimOrderDetails ADD CONSTRAINT
   FK_fudgecorporation_DimOrderDetails_ProductID FOREIGN KEY
   (
   ProductID
   ) REFERENCES fudgecorporation.DimProducts
   ( ProductKey )
     ON UPDATE  NO ACTION
     ON DELETE  NO ACTION
;
------------------------------------------------------------------------------ Fudge Corp Sales Business Process --------------------------------------------------------------------------------------------------------------------------
/* Drop table dbo.FactSales */
IF EXISTS (SELECT * FROM dbo.sysobjects WHERE id = OBJECT_ID(N'dbo.FactSales') AND OBJECTPROPERTY(id, N'IsUserTable') = 1)
DROP TABLE fudgecorporation.FactSales 
;

/* Create table dbo.FactSales */
CREATE TABLE fudgecorporation.FactSales (
   [ProductKey]  int  NOT NULL
,  [CustomerKey]  int NOT NULL
,  [OrderKey]  int  NOT NULL
,  [OrderDate]  int  NOT NULL
,  [CompanyKey]  int   NOT NULL
,  [OrderQuantity]  int   NOT NULL
,  [RetailPrice]  money   NOT NULL
,  [RetailRevenue]  money   NOT NULL
, CONSTRAINT [PK_fudgecorporation.FactSales] PRIMARY KEY NONCLUSTERED 
( [ProductKey], [OrderKey], [CompanyKey], [CustomerKey])
) ON [PRIMARY]
;

ALTER TABLE fudgecorporation.FactSales ADD CONSTRAINT
   FK_fudgecorporation_FactSales_ProductKey FOREIGN KEY
   (
   ProductKey
   ) REFERENCES fudgecorporation.DimProducts
   ( ProductKey )
     ON UPDATE  NO ACTION
     ON DELETE  NO ACTION
;
 
ALTER TABLE fudgecorporation.FactSales ADD CONSTRAINT
   FK_fudgecorporation_FactSales_CustomerKey FOREIGN KEY
   (
   CustomerKey
   ) REFERENCES fudgecorporation.DimCustomers
   ( CustomerKey )
     ON UPDATE  NO ACTION
     ON DELETE  NO ACTION
;
 
ALTER TABLE fudgecorporation.FactSales ADD CONSTRAINT
   FK_fudgecorporation_FactSales_OrderKey FOREIGN KEY
   (
   OrderKey
   ) REFERENCES fudgecorporation.DimOrders
   ( OrderKey )
     ON UPDATE  NO ACTION
     ON DELETE  NO ACTION
;
 
ALTER TABLE fudgecorporation.FactSales ADD CONSTRAINT
   FK_fudgecorporation_FactSales_OrderDate FOREIGN KEY
   (
   OrderDate
   ) REFERENCES fudgecorporation.DimDate
   ( DateKey )
     ON UPDATE  NO ACTION
     ON DELETE  NO ACTION
;
 
ALTER TABLE fudgecorporation.FactSales ADD CONSTRAINT
   FK_fudgecorporation_FactSales_CompanyKey FOREIGN KEY
   (
   CompanyKey
   ) REFERENCES fudgecorporation.DimCompany
   ( CompanyKey )
     ON UPDATE  NO ACTION
     ON DELETE  NO ACTION
;


------------------------------------------------------------------------------ Employee Costs Business Process --------------------------------------------------------------------------------------------------------------------------
/* Create table fudgecorporation.FactEmployeeExpense */
CREATE TABLE fudgecorporation.FactEmployeeExpenses (
   [EmployeeKey]  int  NOT NULL
,  [PayrollDate]  int NOT NULL
,  [HourlyRate]  money  NOT NULL
,  [Hours]  decimal(3,1)  NOT NULL
,  [Cost]  money   NOT NULL
,  [CompanyKey] int  NOT NULL
, CONSTRAINT [PK_fudgecorporation.FactEmployeeExpenses] PRIMARY KEY NONCLUSTERED 
( [EmployeeKey], [PayrollDate], [CompanyKey])
) ON [PRIMARY]
;

ALTER TABLE fudgecorporation.FactEmployeeExpenses ADD CONSTRAINT
   FK_fudgecorporation_FactEmployeeExpenses_EmployeeKey FOREIGN KEY
   (
   EmployeeKey
   ) REFERENCES fudgecorporation.DimEmployees
   ( EmployeeKey )
     ON UPDATE  NO ACTION
     ON DELETE  NO ACTION
;

ALTER TABLE fudgecorporation.FactEmployeeExpenses ADD CONSTRAINT
   FK_fudgecorporation_FactEmployeeExpenses_CompanyKey FOREIGN KEY
   (
   CompanyKey
   ) REFERENCES fudgecorporation.DimCompany
   ( CompanyKey )
     ON UPDATE  NO ACTION
     ON DELETE  NO ACTION
;

ALTER TABLE fudgecorporation.FactEmployeeExpenses ADD CONSTRAINT
   FK_fudgecorporation_FactEmployeeExpenses_PayrollDate FOREIGN KEY
   (
   PayrollDate
   ) REFERENCES fudgecorporation.DimDate
   ( DateKey )
     ON UPDATE  NO ACTION
     ON DELETE  NO ACTION
;