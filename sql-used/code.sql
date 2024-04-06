use ContosoRetailDW

select
	concat(FirstName, ' ', LastName) as CustomerName,
	datediff(YEAR, BirthDate, GETDATE()) as Age,
	MaritalStatus,
	Gender,
	YearlyIncome,
	TotalChildren,
	NumberChildrenAtHome,
	Education,
	Occupation,
	HouseOwnerFlag,
	NumberCarsOwned,
	CustomerType,
	case
		when TotalChildren <= (YearlyIncome * 0.00002) and HouseOwnerFlag <= (YearlyIncome * 0.00008) and NumberCarsOwned <= (YearlyIncome * 0.00004) then 'Good'
		when TotalChildren <= (YearlyIncome * 0.00006) and HouseOwnerFlag < (YearlyIncome * 0.00006) and NumberCarsOwned < (YearlyIncome * 0.00006) then 'Standard'
		else 'Poor'
		end as CustomerScore
from DimCustomer
where CustomerType = 'Person'
