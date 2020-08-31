import pickle
import numpy as np
reg = pickle.load(open('rf_regressor_gridsearch.pkl', 'rb'))

n_quartos = input('entre com o numero de quartos')
area = input('entre com a area')
zona = input('entre com a zona')

zona_to_onehot = {'leste': np.array([1, 0, 0, 0]),
                  'norte': np.array([0, 1, 0, 0]),
                  'oeste': np.array([0, 0, 1, 0]),
                  'sul'  : np.array([0, 0, 0, 1])
                  }

n_quartos_log = np.log1p(int(n_quartos))
area_log = np.log1p(float(area))
zona_onehot = zona_to_onehot[zona]

pred=np.expm1(reg.predict([np.r_[zona_onehot, n_quartos_log, area_log]]))
print(pred)
