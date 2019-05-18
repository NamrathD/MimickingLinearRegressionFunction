# MimickingLinearRegressionFunction
I built a Linear Regression Function from scratch to get good understanding of it and its characteristics.

Some important lessons I learned from creating a class that mimics a linear regression function.


Importance of value of alpha

I learned practically how important choosing the right alpha was to get the right result. For one of the data examples, when I set the alpha to 0.00001 the parameters kept on increasing and went towards infinity. At first, I was confused and doubtful on my program logic, but as soon as I changed the alpha to 0.000001 vs previous 0.00001, parameters started to get more of an accurate results. Although it was just one decimal place difference, it changed the output of the program entirely. 


Importance of # of iterations and alpha

I practically learned what type of mutual relation # of iterations and alpha have. I learned that # of iterations are inversely proportional to value of alpha. The lower the alpha (which generally produces more precise result compared to higher alpha), the more number of iterations required to get to the accurate result.


Importance of feature scaling

After hours of headache of trying to get the right parameters for a specific data examples, I realized how necessary it is to scale the features to make it easier to find the right parameters. Scaling the features decreases the importance of choosing the value of alpha to a minute precision.
