Telco Churn Drivers

#*Project Description*


The Telco dataset was initially created by IBM so that individuals and other interested parties could have a somewhat realistic dataset to practice data analysis on. The customers each have different characteristics which include things like whether or not they’re a senior citizen, have a partner or dependents, how they pay their bill, what services they have and whether or not they have churned (if they are still a customer or have left).

*Project Goal*


Discover drivers of customer churn in the Telco dataset
Use those drivers to develop machine learning model(s) to classify whether or not a customer is likely to churn or not
Churn is defined as a customer who is no longer a Telco customer
This information can be used to develop insights and incentives for customers to stay with the company

*Initial Thoughts*


I believe that some combination of contract type, and billing amount will have the greatest influence on churn.
Possibly the internet service type could affect churn, customers may not be happy with one type of service. 

*The Plan*


Acquire data from codeup database

*Prepare the data*


-Drop duplicate columns

-Codify variables as numbers if they aren’t already

-Make dollar amounts into floats

-Clear whitespace and null values


*Explore the data for drivers* 


Answer these initial questions:

-Do customers with more or less expensive Total bills maintain their accounts for longer periods of time?

-Does payment type have any kind of correlation to what services a customer has?

-What contract type has the most churn?

-Does internet service type correlate to churn likelihood?

-Does a customer’s monthly bill (not total charges) affect churn likelihood?



*Develop a model to predict if a customer is likely to churn or not*

Use driver(s) identified in explore to create a predictive model, and make models for each type of classifier
Evaluate the models on train and validate data splits
Select the best model based on which has the highest accuracy
Evaluate the chosen model on test data


*Data Dictionary - Term*         Definition
-Gender
Male or Female, codified as 0 and 1 respectively

-Senior Citizen
Whether or not a customer is 65 years old, 0 = No, 1 = Yes

-Partner
If a customer has a partner living with them, 0 = No, 1 = Yes

-Dependents
If a customer has dependents living with them, 0 = No, 1 = Yes

-Tenure
The length of time in months a customer has been with Telco

-Phone Service
If a customer has phone service, 0 = No, 1 = Yes

-Multiple Lines
If a customer has multiple phone lines, 0 = No, 1 = Yes

-Online Security
If a customer has paid Online Security service through Telco, 0 = No, 1 = Yes

-Online Backup
If a customer has paid Online Backup service through Telco, 0 = No, 1 = Yes

-Device Protection
If a customer has paid Device Protection service through Telco, 0 = No, 1 = Yes

-Tech Support
If a customer has paid Tech Support service through Telco, 0 = No, 1 = Yes

-Streaming TV
If a customer has Streaming TV service, 0 = No, 1 = Yes

-Streaming Movies
If a customer has Streaming Movie service, 0 = No, 1 = Yes

-Paperless Billing
If a customer has signed up for Paperless Billing, 0 = No, 1 = Yes

-Monthly Charges
The Monthly bill a customer receives in dollars

-Total Charges
The Total Charges in dollars a customer has paid while they have been a Telco customer

-Churn
Whether or not a customer is still with Telco

-Contract Type
Month-to-Month, One Year, Two Year contracts codified as 0, 1, 2 respectively

-Internet Service Type
No internet, FiberOptic, DSL, codified as 0, 1, 2 respectively

-Payment Type
Mailed check, Electronic Check, Bank Transfer (automatic), Credit Card (automatic)
Codified as 0, 1, 2, 3 respectively



*Steps to Reproduce*

1.Clone this Reposition: Presentation and Acquire should be the only needed files
2.Make sure the files are in the same location and run the presentation notebook
3.Profit?


*Takeaways and Conclusions*


   -The main driver we identified with chi2 significance was the total charges that a customer paid over their tenure. This could be interpreted to mean that customer retention might increase if Telco was able to provide discounts to these high-bill customers, or incentives for their long tenure (and customer loyalty) like increased speeds for the same price bracket, free TV/Movie streaming, or other services they could easily enable based on a customer’s total spend. 


   -The other chosen characteristics may not have has chi2 significance, but the visualizations created for their comparisons certainly identified areas where smaller changes may lead to increased customer satisfaction and retention; for example, FiberOptic certainly has more churned customers than DSL, perhaps because the service is not as reliable as DSL or other providers. Telco’s focus on increasing reliability could shore up customer losses. 


   -Month-to-month customers have the easiest contracts to drop, so incentivizing a lower cost contract for a year and providing benefits for signing up for a two-year contract could be easy ways to convert month-to-month customers to other, longer, contract types. 


*Recommendations*
  I would like to examine a larger spread and crosstabulation of a larger number of variables.
  Perhaps its possible that being partnerless and a senior citizen is the most reliable customer
  type, or maybe they’re the most likely to shop around for the most effective use of their
  dollars. I’m sure there are a few unexpected correlations that would be interesting to discover. 
