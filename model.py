import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

no_trend_x = pd.DataFrame(no_trend_x)
no_trend_x.index.freq = "MS"

# Split into train and test set
X_train_no_trend = no_trend_x[:120]
X_test_no_trend = no_trend_x[120:]

#treinar o modelo
model = ExponentialSmoothing(X_train_no_trend[0], 
                             trend="add", seasonal="add", 
                             seasonal_periods=5).fit()
#predizer observações
Y_HW_no_trend = model.forecast(len(X_test_no_trend[0]))