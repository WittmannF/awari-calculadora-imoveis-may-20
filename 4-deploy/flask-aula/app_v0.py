import pickle
import numpy as np
reg = pickle.load(open('rf_regressor_gridsearch.pkl', 'rb'))

pred=np.expm1(reg.predict([[0, 0, 0, 1, np.log1p(2), np.log1p(120)]]))
print(pred)
