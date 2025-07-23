--drop view acs_housing_vw

create or replace view acs_housing_vw as
select 
	p.*, 
	gcz.zip,
	gcz.primary_city, 
	gcz.county, 
	gcz.latitude, 
	gcz.longitude
from acs_housing_final p 
inner join geo_corr_zip gcz 
	on gcz.puma22::int = p.puma_normalized::int
	order by 1
	
	