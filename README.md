#RandomForestFunctions
RandomForestFunctions is a way to functionalize the process of classical machine learning with the random forest classifier. 

etl.py and mltool.py are to be used hand in hand.

etl.py transfroms Pandas DataFrames to numpy arrays and back.
It also one-hot encodes categorical data and stores labels and features so they are not lost in the transformation.

mltool.py helps users understand their dataset and validate their data as well.
mltool.py can compute feature importances and cross validate models.

