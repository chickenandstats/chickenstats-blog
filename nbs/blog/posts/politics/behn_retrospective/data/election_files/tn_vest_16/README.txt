2016 Tennessee precinct and election results shapefile.

## RDH Date retrieval
01/20/2022

## Sources
Election results from the Tennessee Secretary of State (https://sos.tn.gov/elections/results#2016).
Precinct shapefile primarily from the Tennessee Comptroller of the Treasury (https://apps.cot.tn.gov/DPAMaps/Redistrict/Counties).
Hamilton County replaced with a shapefile from the county GIS department.

## Fields metadata

Vote Column Label Format
------------------------
Columns reporting votes follow a standard label pattern. One example is:
G16PREDCli
The first character is G for a general election, P for a primary, C for a caucus, R for a runoff, S for a special.
Characters 2 and 3 are the year of the election.
Characters 4-6 represent the office type (see list below).
Character 7 represents the party of the candidate.
Characters 8-10 are the first three letters of the candidate's last name.

Office Codes
AGR - Commissioner of Agriculture
ATG - Attorney General
AUD - Auditor
COM - Comptroller
COU - City Council Member
DEL - Delegate to the U.S. House
GOV - Governor
H## - U.S. House, where ## is the district number. AL: at large.
HOD - House of Delegates, accompanied by a HOD_DIST column indicating district number
HOR - U.S. House, accompanied by a HOR_DIST column indicating district number
INS - Commissioner of Insurance
LAB - Commissioner of Labor
LTG - Lieutenant Governor
LND - Commissioner of Public Lands
PRE - President
PSC - Public Service Commissioner
PUC - Public Utilities Commissioner
RGT - State University Regent
RRC - Railroad Commissioner
SAC - State Court of Appeals
SCC - State Court of Criminal Appeals
SOS - Secretary of State
SOV - Senate of Virginia, accompanied by a SOV_DIST column indicating district number
SPI - Superintendent of Public Instruction
SSC - State Supreme Court
TRE - Treasurer
USS - U.S. Senate

Party Codes
D and R will always represent Democrat and Republican, respectively.
See the state-specific notes for the remaining codes used in a particular file; note that third-party candidates may appear on the ballot under different party labels in different states.

## Fields
G16PRERTRU - Donald J. Trump (Republican Party)
G16PREDCLI - Hillary Clinton (Democratic Party)
G16PRELJOH - Gary Johnson (Libertarian Party)
G16PREGSTE - Jill Stein (Green Party)
G16PREISMI - Mike Smith (Independent)
G16PREIKEN - Alyson Kennedy (Independent)
G16PREIFUE - Roque De La Fuente (Independent)
G16PREOWRI - Write-in Votes

## Processing Steps
Davidson and Wilson reported absentee votes countywide. Robertson, Stewart, Union, and Wilson reported provisional votes countywide. These were distributed by candidate to precincts based on their share of the precinct-level reported vote.

Thirty counties reported early vote write-ins countywide. These were similarly distributed to precincts based on their share of the precinct-level reported write-in vote.

In Chester County the East Chester and West Chester precincts were split by the Henderson city limits since county and city votes were reported separately for the 2016 general election.
