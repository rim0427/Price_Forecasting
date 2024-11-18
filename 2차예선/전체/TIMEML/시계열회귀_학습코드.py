import random
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import itertools
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_absolute_error
import optuna
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error
import numpy as np
from sklearn.metrics import mean_absolute_error
import numpy as np
import lightgbm as lgb
from FE.common_FE import common_FE
import re
import json


seed = 42
np.random.seed(seed)
random.seed(seed)

sub_test_1 = pd.read_csv('test_1_v2.csv',encoding='cp949')
sub_test_2 = pd.read_csv('test_2_v2.csv',encoding='cp949')

group1 = ['배추', '무', '양파', '감자 수미', '대파(일반)']
group2 = ['건고추', '깐마늘(국산)','상추', '사과', '배']

category = ["감자 수미", "무", "양파", "배추", "대파(일반)", "건고추", "깐마늘(국산)", "사과", "배", "상추"]

best_params_all = {
    "감자 수미": {
        "1순": {
            2018: {
                9555: {
                    "n_estimators": 966,
                    "max_depth": 3,
                    "num_leaves": 105,
                    "learning_rate": 0.08096703476149342,
                    "min_child_samples": 12,
                    "min_split_gain": 0.8915556977465369,
                    "colsample_bytree": 0.7888999170798527,
                    "subsample": 0.9871958105273594,
                    "reg_alpha": 8.404835907651286,
                    "reg_lambda": 3.7604096032174756,
                    "random_state": 9555
                }
            },
            2019: {
                9555: {
                    "n_estimators": 2218,
                    "max_depth": 4,
                    "num_leaves": 80,
                    "learning_rate": 0.12797383189117717,
                    "min_child_samples": 8,
                    "min_split_gain": 0.06907616252289671,
                    "colsample_bytree": 0.7018165374360806,
                    "subsample": 0.9543103341644905,
                    "reg_alpha": 0.00538216342905961,
                    "reg_lambda": 7.734383409234123,
                    "random_state": 9555
                }
            },
            2020: {
                9555: {
                    "n_estimators": 1418,
                    "max_depth": 7,
                    "num_leaves": 126,
                    "learning_rate": 0.21402204925330856,
                    "min_child_samples": 10,
                    "min_split_gain": 0.8088626522630953,
                    "colsample_bytree": 0.26918381106730177,
                    "subsample": 0.9347144580469737,
                    "reg_alpha": 6.737672650377978,
                    "reg_lambda": 3.461685418776888,
                    "random_state": 9555
                }
            },
            2021: {
                9555: {
                    "n_estimators": 1490,
                    "max_depth": 16,
                    "num_leaves": 69,
                    "learning_rate": 0.010470849918048634,
                    "min_child_samples": 16,
                    "min_split_gain": 0.5599734555932082,
                    "colsample_bytree": 0.18351576823500204,
                    "subsample": 0.9698746758866092,
                    "reg_alpha": 3.053491823830987,
                    "reg_lambda": 4.231810443647769,
                    "random_state": 9555
                }
            },
            2022: {
                9555: {
                    "n_estimators": 1886,
                    "max_depth": 3,
                    "num_leaves": 96,
                    "learning_rate": 0.16653247577596095,
                    "min_child_samples": 5,
                    "min_split_gain": 7.952318239383416e-05,
                    "colsample_bytree": 0.8589721379333327,
                    "subsample": 0.9211405945802689,
                    "reg_alpha": 7.301627396837099,
                    "reg_lambda": 9.970529553027331,
                    "random_state": 9555
                }
            }
        },
        "2순": {
            2018: {
                9555: {
                    "n_estimators": 617,
                    "max_depth": 9,
                    "num_leaves": 131,
                    "learning_rate": 0.08689855687231544,
                    "min_child_samples": 10,
                    "min_split_gain": 0.7072709855246634,
                    "colsample_bytree": 0.30039358602199384,
                    "subsample": 0.6647598132523339,
                    "reg_alpha": 9.277133782033527,
                    "reg_lambda": 8.280206960631954,
                    "random_state": 9555
                }
            },
            2019: {
                9555: {
                    "n_estimators": 625,
                    "max_depth": 19,
                    "num_leaves": 142,
                    "learning_rate": 0.08302342998623788,
                    "min_child_samples": 8,
                    "min_split_gain": 0.8334833277037415,
                    "colsample_bytree": 0.5401111310726305,
                    "subsample": 0.7768244571888416,
                    "reg_alpha": 3.6783916311109315,
                    "reg_lambda": 2.5804937482644523,
                    "random_state": 9555
                }
            },
            2020: {
                9555: {
                    "n_estimators": 1623,
                    "max_depth": 19,
                    "num_leaves": 124,
                    "learning_rate": 0.05947174006501538,
                    "min_child_samples": 8,
                    "min_split_gain": 0.8124520197132274,
                    "colsample_bytree": 0.1458601215024355,
                    "subsample": 0.8802169836170387,
                    "reg_alpha": 6.591369227072848,
                    "reg_lambda": 3.599951621716323,
                    "random_state": 9555
                }
            },
            2021: {
                9555: {
                    "n_estimators": 2348,
                    "max_depth": 3,
                    "num_leaves": 93,
                    "learning_rate": 0.1307562329301655,
                    "min_child_samples": 18,
                    "min_split_gain": 0.007740541787473021,
                    "colsample_bytree": 0.5294722950076431,
                    "subsample": 0.8598261174171505,
                    "reg_alpha": 7.0780490961727285,
                    "reg_lambda": 3.5469963094290824,
                    "random_state": 9555
                }
            },
            2022: {
                9555: {
                    "n_estimators": 535,
                    "max_depth": 11,
                    "num_leaves": 40,
                    "learning_rate": 0.11945717811639668,
                    "min_child_samples": 5,
                    "min_split_gain": 0.7545898261983273,
                    "colsample_bytree": 0.42539701735482593,
                    "subsample": 0.8938738607590799,
                    "reg_alpha": 9.896682978166885,
                    "reg_lambda": 6.026812596993813,
                    "random_state": 9555
                }
            }
        },
        "3순": {
            2018: {
                9555: {
                    "n_estimators": 1789,
                    "max_depth": 13,
                    "num_leaves": 127,
                    "learning_rate": 0.04042661313753281,
                    "min_child_samples": 9,
                    "min_split_gain": 0.5599513003484191,
                    "colsample_bytree": 0.7967690719163315,
                    "subsample": 0.5372990312324226,
                    "reg_alpha": 9.238491110963048,
                    "reg_lambda": 0.8341865035596818,
                    "random_state": 9555
                }
            },
            2019: {
                9555: {
                    "n_estimators": 975,
                    "max_depth": 6,
                    "num_leaves": 107,
                    "learning_rate": 0.12912448166681728,
                    "min_child_samples": 9,
                    "min_split_gain": 0.626233098424595,
                    "colsample_bytree": 0.9801487235735382,
                    "subsample": 0.9999090946293865,
                    "reg_alpha": 3.5329373245356366,
                    "reg_lambda": 2.8466129253155077,
                    "random_state": 9555
                }
            },
            2020: {
                9555: {
                    "n_estimators": 1965,
                    "max_depth": 11,
                    "num_leaves": 95,
                    "learning_rate": 0.1529416210479758,
                    "min_child_samples": 15,
                    "min_split_gain": 0.6692449332911827,
                    "colsample_bytree": 0.9913982884770711,
                    "subsample": 0.5715479452091492,
                    "reg_alpha": 7.244016345056692,
                    "reg_lambda": 7.114610812666042,
                    "random_state": 9555
                }
            },
            2021: {
                9555: {
                    "n_estimators": 2288,
                    "max_depth": 14,
                    "num_leaves": 140,
                    "learning_rate": 0.09291410822655738,
                    "min_child_samples": 7,
                    "min_split_gain": 0.9355382723112073,
                    "colsample_bytree": 0.523957467303895,
                    "subsample": 0.9789984642729105,
                    "reg_alpha": 2.642985724370683,
                    "reg_lambda": 6.227883589279358,
                    "random_state": 9555
                }
            },
            2022: {
                9555: {
                    "n_estimators": 2223,
                    "max_depth": 3,
                    "num_leaves": 55,
                    "learning_rate": 0.0673340680646652,
                    "min_child_samples": 8,
                    "min_split_gain": 0.5366033884528563,
                    "colsample_bytree": 0.7529110202247584,
                    "subsample": 0.8678192506260854,
                    "reg_alpha": 4.341269832022734,
                    "reg_lambda": 5.131899864673432,
                    "random_state": 9555
                }
            }
        }
    },
    "무": {
        "1순": {
            2018: {
                9555: {
                    "n_estimators": 632,
                    "max_depth": 5,
                    "num_leaves": 43,
                    "learning_rate": 0.29187783202412243,
                    "min_child_samples": 36,
                    "min_split_gain": 0.008015723156835103,
                    "colsample_bytree": 0.9787817943178545,
                    "subsample": 0.6304243055589878,
                    "reg_alpha": 7.691664195779399,
                    "reg_lambda": 5.733245296314868,
                    "random_state": 9555
                }
            },
            2019: {
                9555: {
                    "n_estimators": 672,
                    "max_depth": 14,
                    "num_leaves": 145,
                    "learning_rate": 0.22547301743189987,
                    "min_child_samples": 16,
                    "min_split_gain": 0.1442639685044536,
                    "colsample_bytree": 0.9035819405428756,
                    "subsample": 0.6745188210792561,
                    "reg_alpha": 7.89254745991273,
                    "reg_lambda": 1.78271205131996,
                    "random_state": 9555
                }
            },
            2020: {
                9555: {
                    "n_estimators": 1634,
                    "max_depth": 3,
                    "num_leaves": 36,
                    "learning_rate": 0.050184865570187295,
                    "min_child_samples": 10,
                    "min_split_gain": 0.49210150125013247,
                    "colsample_bytree": 0.27339570419858805,
                    "subsample": 0.6334805538941698,
                    "reg_alpha": 1.5300587081722128,
                    "reg_lambda": 9.857201417166786,
                    "random_state": 9555
                }
            },
            2021: {
                9555: {
                    "n_estimators": 419,
                    "max_depth": 4,
                    "num_leaves": 120,
                    "learning_rate": 0.08855487304483303,
                    "min_child_samples": 32,
                    "min_split_gain": 0.2051225673982785,
                    "colsample_bytree": 0.1497273550966243,
                    "subsample": 0.5361404404348786,
                    "reg_alpha": 1.196702617915785,
                    "reg_lambda": 7.522922992287484,
                    "random_state": 9555
                }
            },
            2022: {
                9555: {
                    "n_estimators": 1275,
                    "max_depth": 9,
                    "num_leaves": 104,
                    "learning_rate": 0.07203105613237255,
                    "min_child_samples": 41,
                    "min_split_gain": 0.6788225288313567,
                    "colsample_bytree": 0.19396290476106942,
                    "subsample": 0.7216308250233257,
                    "reg_alpha": 6.014659018474033,
                    "reg_lambda": 5.053673449125142,
                    "random_state": 9555
                }
            }
        },
        "2순": {
            2018: {
                9555: {
                    "n_estimators": 1855,
                    "max_depth": 20,
                    "num_leaves": 135,
                    "learning_rate": 0.06983684530726117,
                    "min_child_samples": 46,
                    "min_split_gain": 0.4165331171959163,
                    "colsample_bytree": 0.34750025084275377,
                    "subsample": 0.943317930330767,
                    "reg_alpha": 7.018984463062559,
                    "reg_lambda": 3.4065815002181874,
                    "random_state": 9555
                }
            },
            2019: {
                9555: {
                    "n_estimators": 1515,
                    "max_depth": 10,
                    "num_leaves": 51,
                    "learning_rate": 0.2907114493667321,
                    "min_child_samples": 12,
                    "min_split_gain": 0.056785804180405214,
                    "colsample_bytree": 0.6851822128612443,
                    "subsample": 0.6576328082197683,
                    "reg_alpha": 4.5817057839355115,
                    "reg_lambda": 6.1036668934313445,
                    "random_state": 9555
                }
            },
            2020: {
                9555: {
                    "n_estimators": 561,
                    "max_depth": 7,
                    "num_leaves": 82,
                    "learning_rate": 0.03855952691397907,
                    "min_child_samples": 26,
                    "min_split_gain": 0.0029393356804219926,
                    "colsample_bytree": 0.18131903159976093,
                    "subsample": 0.5865212775743172,
                    "reg_alpha": 8.655033657548854,
                    "reg_lambda": 9.48952585321216,
                    "random_state": 9555
                }
            },
            2021: {
                9555: {
                    "n_estimators": 1725,
                    "max_depth": 10,
                    "num_leaves": 66,
                    "learning_rate": 0.08032743261726866,
                    "min_child_samples": 6,
                    "min_split_gain": 0.33593070560178856,
                    "colsample_bytree": 0.10155939654336854,
                    "subsample": 0.5039873408303291,
                    "reg_alpha": 9.992603451175185,
                    "reg_lambda": 0.23981503364160517,
                    "random_state": 9555
                }
            },
            2022: {
                9555: {
                    "n_estimators": 1523,
                    "max_depth": 9,
                    "num_leaves": 137,
                    "learning_rate": 0.046249357231541305,
                    "min_child_samples": 40,
                    "min_split_gain": 0.6531050933272217,
                    "colsample_bytree": 0.17125679617750972,
                    "subsample": 0.9319228225384534,
                    "reg_alpha": 2.8774449946106495,
                    "reg_lambda": 9.749641031993965,
                    "random_state": 9555
                }
            }
        },
        "3순": {
            2018: {
                9555: {
                    "n_estimators": 2254,
                    "max_depth": 12,
                    "num_leaves": 95,
                    "learning_rate": 0.24056313387123976,
                    "min_child_samples": 24,
                    "min_split_gain": 0.08048376307137663,
                    "colsample_bytree": 0.44904989526297234,
                    "subsample": 0.7958830973895668,
                    "reg_alpha": 9.977624902217467,
                    "reg_lambda": 6.509429407226579,
                    "random_state": 9555
                }
            },
            2019: {
                9555: {
                    "n_estimators": 857,
                    "max_depth": 19,
                    "num_leaves": 115,
                    "learning_rate": 0.01958312539628232,
                    "min_child_samples": 63,
                    "min_split_gain": 0.08039138728316045,
                    "colsample_bytree": 0.7781334370842309,
                    "subsample": 0.9715737745962507,
                    "reg_alpha": 5.590991124907465,
                    "reg_lambda": 4.978271738958599,
                    "random_state": 9555
                }
            },
            2020: {
                9555: {
                    "n_estimators": 1865,
                    "max_depth": 17,
                    "num_leaves": 114,
                    "learning_rate": 0.15654566469477177,
                    "min_child_samples": 54,
                    "min_split_gain": 0.8460596935657385,
                    "colsample_bytree": 0.1575176171036068,
                    "subsample": 0.8613352223219484,
                    "reg_alpha": 4.843603444764613,
                    "reg_lambda": 5.911568784799486,
                    "random_state": 9555
                }
            },
            2021: {
                9555: {
                    "n_estimators": 489,
                    "max_depth": 10,
                    "num_leaves": 54,
                    "learning_rate": 0.028569677912327934,
                    "min_child_samples": 64,
                    "min_split_gain": 0.15824271792198638,
                    "colsample_bytree": 0.23273170660977446,
                    "subsample": 0.7771946662665429,
                    "reg_alpha": 2.0650915592713455,
                    "reg_lambda": 3.8243631602530814,
                    "random_state": 9555
                }
            },
            2022: {
                9555: {
                    "n_estimators": 958,
                    "max_depth": 18,
                    "num_leaves": 95,
                    "learning_rate": 0.2751922047900801,
                    "min_child_samples": 25,
                    "min_split_gain": 0.3635653605421259,
                    "colsample_bytree": 0.21444773687483448,
                    "subsample": 0.969112353072202,
                    "reg_alpha": 1.9310973518217631,
                    "reg_lambda": 9.830374429169785,
                    "random_state": 9555
                }
            }
        }
    },
    "양파": {
        "1순": {
            2018: {
                9555: {
                    "n_estimators": 1989,
                    "max_depth": 3,
                    "num_leaves": 82,
                    "learning_rate": 0.011553995020024042,
                    "min_child_samples": 7,
                    "min_split_gain": 0.6718849572615455,
                    "colsample_bytree": 0.22358980565988462,
                    "subsample": 0.9077353815121828,
                    "reg_alpha": 8.606372348677807,
                    "reg_lambda": 9.9289502598115,
                    "random_state": 9555
                }
            },
            2019: {
                9555: {
                    "n_estimators": 2568,
                    "max_depth": 4,
                    "num_leaves": 121,
                    "learning_rate": 0.03846002547018677,
                    "min_child_samples": 5,
                    "min_split_gain": 0.9778582237573424,
                    "colsample_bytree": 0.1969704091683336,
                    "subsample": 0.8153792407580128,
                    "reg_alpha": 9.0174949451467,
                    "reg_lambda": 6.431207096201219,
                    "random_state": 9555
                }
            },
            2020: {
                9555: {
                    "n_estimators": 2373,
                    "max_depth": 14,
                    "num_leaves": 122,
                    "learning_rate": 0.18568810800321053,
                    "min_child_samples": 9,
                    "min_split_gain": 0.10762438404850205,
                    "colsample_bytree": 0.13731846986050852,
                    "subsample": 0.7050328369283801,
                    "reg_alpha": 1.4608063202325745,
                    "reg_lambda": 6.370012644058218,
                    "random_state": 9555
                }
            },
            2021: {
                9555: {
                    "n_estimators": 2920,
                    "max_depth": 16,
                    "num_leaves": 24,
                    "learning_rate": 0.05831362703370126,
                    "min_child_samples": 66,
                    "min_split_gain": 0.16448475134784626,
                    "colsample_bytree": 0.7834688787169928,
                    "subsample": 0.5929824932276514,
                    "reg_alpha": 4.265517132126874,
                    "reg_lambda": 1.5958524462658392,
                    "random_state": 9555
                }
            },
            2022: {
                9555: {
                    "n_estimators": 2005,
                    "max_depth": 10,
                    "num_leaves": 111,
                    "learning_rate": 0.2072202197394116,
                    "min_child_samples": 27,
                    "min_split_gain": 0.2744131704006252,
                    "colsample_bytree": 0.47240028014447205,
                    "subsample": 0.9083819574921355,
                    "reg_alpha": 1.5177501936008149,
                    "reg_lambda": 2.7056798066881353,
                    "random_state": 9555
                }
            }
        },
        "2순": {
            2018: {
                9555: {
                    "n_estimators": 492,
                    "max_depth": 17,
                    "num_leaves": 125,
                    "learning_rate": 0.1668453515276357,
                    "min_child_samples": 6,
                    "min_split_gain": 0.08141086245851092,
                    "colsample_bytree": 0.24466587446003224,
                    "subsample": 0.9248432730162495,
                    "reg_alpha": 3.2794558304935837,
                    "reg_lambda": 9.642231750835139,
                    "random_state": 9555
                }
            },
            2019: {
                9555: {
                    "n_estimators": 897,
                    "max_depth": 10,
                    "num_leaves": 43,
                    "learning_rate": 0.020131279772128995,
                    "min_child_samples": 46,
                    "min_split_gain": 0.2739575299832459,
                    "colsample_bytree": 0.29310425629956016,
                    "subsample": 0.9406900406647805,
                    "reg_alpha": 8.659416948705227,
                    "reg_lambda": 1.1433345569209168,
                    "random_state": 9555
                }
            },
            2020: {
                9555: {
                    "n_estimators": 2567,
                    "max_depth": 15,
                    "num_leaves": 33,
                    "learning_rate": 0.13181826288281956,
                    "min_child_samples": 32,
                    "min_split_gain": 0.6642136535938976,
                    "colsample_bytree": 0.13194658241719842,
                    "subsample": 0.9283205387931448,
                    "reg_alpha": 7.74241870476558,
                    "reg_lambda": 0.7893773309151931,
                    "random_state": 9555
                }
            },
            2021: {
                9555: {
                    "n_estimators": 455,
                    "max_depth": 5,
                    "num_leaves": 113,
                    "learning_rate": 0.023332715063230314,
                    "min_child_samples": 60,
                    "min_split_gain": 0.18116809280288837,
                    "colsample_bytree": 0.5187587838523994,
                    "subsample": 0.8456625433690795,
                    "reg_alpha": 4.559729652256554,
                    "reg_lambda": 3.0146937900962554,
                    "random_state": 9555
                }
            },
            2022: {
                9555: {
                    "n_estimators": 1907,
                    "max_depth": 5,
                    "num_leaves": 83,
                    "learning_rate": 0.2611478630599706,
                    "min_child_samples": 20,
                    "min_split_gain": 0.8941613660331341,
                    "colsample_bytree": 0.9977616918542624,
                    "subsample": 0.9979423826430276,
                    "reg_alpha": 3.1397829172633203,
                    "reg_lambda": 4.611879574044535,
                    "random_state": 9555
                }
            }
        },
        "3순": {
            2018: {
                9555: {
                    "n_estimators": 1306,
                    "max_depth": 17,
                    "num_leaves": 56,
                    "learning_rate": 0.2763501515304565,
                    "min_child_samples": 14,
                    "min_split_gain": 0.17686174043361613,
                    "colsample_bytree": 0.18593262492912965,
                    "subsample": 0.7926071707785591,
                    "reg_alpha": 7.89619749754208,
                    "reg_lambda": 5.568769993038821,
                    "random_state": 9555
                }
            },
            2019: {
                9555: {
                    "n_estimators": 319,
                    "max_depth": 20,
                    "num_leaves": 82,
                    "learning_rate": 0.07396327737346033,
                    "min_child_samples": 47,
                    "min_split_gain": 0.20190520765964437,
                    "colsample_bytree": 0.11477675518211189,
                    "subsample": 0.8969588156867343,
                    "reg_alpha": 8.406620899010312,
                    "reg_lambda": 9.550526290635972,
                    "random_state": 9555
                }
            },
            2020: {
                9555: {
                    "n_estimators": 1403,
                    "max_depth": 4,
                    "num_leaves": 33,
                    "learning_rate": 0.18265376839306519,
                    "min_child_samples": 33,
                    "min_split_gain": 0.6824165479699661,
                    "colsample_bytree": 0.1402723582022941,
                    "subsample": 0.9305818755466494,
                    "reg_alpha": 0.9972218455571333,
                    "reg_lambda": 2.213403413611212,
                    "random_state": 9555
                }
            },
            2021: {
                9555: {
                    "n_estimators": 533,
                    "max_depth": 7,
                    "num_leaves": 29,
                    "learning_rate": 0.045062701918168066,
                    "min_child_samples": 63,
                    "min_split_gain": 0.02923242156051975,
                    "colsample_bytree": 0.5902886380439558,
                    "subsample": 0.5068729668722618,
                    "reg_alpha": 3.92213031149299,
                    "reg_lambda": 1.5667409211352195,
                    "random_state": 9555
                }
            },
            2022: {
                9555: {
                    "n_estimators": 2919,
                    "max_depth": 4,
                    "num_leaves": 146,
                    "learning_rate": 0.18850691167184633,
                    "min_child_samples": 13,
                    "min_split_gain": 0.5204762413047518,
                    "colsample_bytree": 0.9451530681142982,
                    "subsample": 0.6865172441109215,
                    "reg_alpha": 1.4134535601949396,
                    "reg_lambda": 2.9592998628177045,
                    "random_state": 9555
                }
            }
        }
    },
    "배추": {
        "1순": {
            2018: {
                9555: {
                    "n_estimators": 2203,
                    "max_depth": 20,
                    "num_leaves": 141,
                    "learning_rate": 0.10254121173820466,
                    "min_child_samples": 8,
                    "min_split_gain": 0.09505650259168766,
                    "colsample_bytree": 0.3240504771272241,
                    "subsample": 0.6935633541970502,
                    "reg_alpha": 0.37186480537682565,
                    "reg_lambda": 9.328562831605394,
                    "random_state": 9555
                }
            },
            2019: {
                9555: {
                    "n_estimators": 2712,
                    "max_depth": 16,
                    "num_leaves": 34,
                    "learning_rate": 0.037664029838453544,
                    "min_child_samples": 10,
                    "min_split_gain": 0.33631306111576204,
                    "colsample_bytree": 0.6067065780185853,
                    "subsample": 0.7446735171956594,
                    "reg_alpha": 5.748453310560069,
                    "reg_lambda": 5.1878046937796185,
                    "random_state": 9555
                }
            },
            2020: {
                9555: {
                    "n_estimators": 537,
                    "max_depth": 18,
                    "num_leaves": 36,
                    "learning_rate": 0.07377697864529234,
                    "min_child_samples": 20,
                    "min_split_gain": 0.36269202386801436,
                    "colsample_bytree": 0.5771477562997864,
                    "subsample": 0.9467098707068795,
                    "reg_alpha": 6.772167701085504,
                    "reg_lambda": 6.157851860734073,
                    "random_state": 9555
                }
            },
            2021: {
                9555: {
                    "n_estimators": 2254,
                    "max_depth": 13,
                    "num_leaves": 54,
                    "learning_rate": 0.06349763355771333,
                    "min_child_samples": 14,
                    "min_split_gain": 0.8269935740962808,
                    "colsample_bytree": 0.19896222605201866,
                    "subsample": 0.8979785074813075,
                    "reg_alpha": 2.2381646228165293,
                    "reg_lambda": 2.5939243668740057,
                    "random_state": 9555
                }
            },
            2022: {
                9555: {
                    "n_estimators": 1602,
                    "max_depth": 16,
                    "num_leaves": 116,
                    "learning_rate": 0.11941153526493423,
                    "min_child_samples": 8,
                    "min_split_gain": 0.39143456100870616,
                    "colsample_bytree": 0.5458700547739916,
                    "subsample": 0.7615113273795362,
                    "reg_alpha": 0.41106673773175584,
                    "reg_lambda": 6.729850045936099,
                    "random_state": 9555
                }
            }
        },
        "2순": {
            2018: {
                9555: {
                    "n_estimators": 1554,
                    "max_depth": 7,
                    "num_leaves": 113,
                    "learning_rate": 0.10851629861812381,
                    "min_child_samples": 12,
                    "min_split_gain": 0.5196281486177514,
                    "colsample_bytree": 0.13772379193887369,
                    "subsample": 0.8041541814082054,
                    "reg_alpha": 1.3877773461694494,
                    "reg_lambda": 4.385494243033685,
                    "random_state": 9555
                }
            },
            2019: {
                9555: {
                    "n_estimators": 2807,
                    "max_depth": 19,
                    "num_leaves": 84,
                    "learning_rate": 0.029088840591759556,
                    "min_child_samples": 12,
                    "min_split_gain": 0.9034210207437722,
                    "colsample_bytree": 0.3547466808976034,
                    "subsample": 0.9187567729152,
                    "reg_alpha": 4.647762070571924,
                    "reg_lambda": 1.6439372333362567,
                    "random_state": 9555
                }
            },
            2020: {
                9555: {
                    "n_estimators": 1557,
                    "max_depth": 16,
                    "num_leaves": 39,
                    "learning_rate": 0.0900439275323126,
                    "min_child_samples": 5,
                    "min_split_gain": 0.08163637388072326,
                    "colsample_bytree": 0.3258624104774232,
                    "subsample": 0.9566550285061052,
                    "reg_alpha": 5.3238697926439835,
                    "reg_lambda": 5.88884329510872,
                    "random_state": 9555
                }
            },
            2021: {
                9555: {
                    "n_estimators": 395,
                    "max_depth": 5,
                    "num_leaves": 91,
                    "learning_rate": 0.07261717140577392,
                    "min_child_samples": 11,
                    "min_split_gain": 0.1608879365174024,
                    "colsample_bytree": 0.11102923137699947,
                    "subsample": 0.9411283938291047,
                    "reg_alpha": 8.805327309476677,
                    "reg_lambda": 3.1530290476666254,
                    "random_state": 9555
                }
            },
            2022: {
                9555: {
                    "n_estimators": 2856,
                    "max_depth": 17,
                    "num_leaves": 77,
                    "learning_rate": 0.07998505999963415,
                    "min_child_samples": 22,
                    "min_split_gain": 0.839163956676582,
                    "colsample_bytree": 0.7202665131030952,
                    "subsample": 0.5392158651573798,
                    "reg_alpha": 0.4017986803710667,
                    "reg_lambda": 9.331814523216607,
                    "random_state": 9555
                }
            }
        },
        "3순": {
            2018: {
                9555: {
                    "n_estimators": 1831,
                    "max_depth": 16,
                    "num_leaves": 51,
                    "learning_rate": 0.29672535807606903,
                    "min_child_samples": 6,
                    "min_split_gain": 0.03135485354517026,
                    "colsample_bytree": 0.9670169087683285,
                    "subsample": 0.5012828826203474,
                    "reg_alpha": 3.835261909997048,
                    "reg_lambda": 6.574521911473265,
                    "random_state": 9555
                }
            },
            2019: {
                9555: {
                    "n_estimators": 1615,
                    "max_depth": 9,
                    "num_leaves": 72,
                    "learning_rate": 0.030323928850146648,
                    "min_child_samples": 7,
                    "min_split_gain": 0.4236175157542418,
                    "colsample_bytree": 0.632336121690524,
                    "subsample": 0.7317336509162419,
                    "reg_alpha": 2.660871846242743,
                    "reg_lambda": 5.966909269687008,
                    "random_state": 9555
                }
            },
            2020: {
                9555: {
                    "n_estimators": 2323,
                    "max_depth": 14,
                    "num_leaves": 95,
                    "learning_rate": 0.015663385399360963,
                    "min_child_samples": 6,
                    "min_split_gain": 0.3808641285506777,
                    "colsample_bytree": 0.10480724047471118,
                    "subsample": 0.6048480023935892,
                    "reg_alpha": 1.747815225071312,
                    "reg_lambda": 2.5320493366552426,
                    "random_state": 9555
                }
            },
            2021: {
                9555: {
                    "n_estimators": 2231,
                    "max_depth": 11,
                    "num_leaves": 104,
                    "learning_rate": 0.08539822645935785,
                    "min_child_samples": 12,
                    "min_split_gain": 0.7883609338256319,
                    "colsample_bytree": 0.15535857319139812,
                    "subsample": 0.7997854825639179,
                    "reg_alpha": 4.3970582704397145,
                    "reg_lambda": 9.645185027618162,
                    "random_state": 9555
                }
            },
            2022: {
                9555: {
                    "n_estimators": 1981,
                    "max_depth": 9,
                    "num_leaves": 24,
                    "learning_rate": 0.296297361632776,
                    "min_child_samples": 24,
                    "min_split_gain": 0.1817760487061083,
                    "colsample_bytree": 0.28468690359766696,
                    "subsample": 0.7969227661072136,
                    "reg_alpha": 9.756693334379285,
                    "reg_lambda": 7.54690013716704,
                    "random_state": 9555
                }
            }
        }
    },
    "대파(일반)": {
        "1순": {
            2018: {
                9555: {
                    "n_estimators": 442,
                    "max_depth": 20,
                    "num_leaves": 103,
                    "learning_rate": 0.25401750634187026,
                    "min_child_samples": 5,
                    "min_split_gain": 0.005295670490468203,
                    "colsample_bytree": 0.9964791166373752,
                    "subsample": 0.9972712355601837,
                    "reg_alpha": 2.9509555542525066,
                    "reg_lambda": 0.022763670574166783,
                    "random_state": 9555
                }
            },
            2019: {
                9555: {
                    "n_estimators": 995,
                    "max_depth": 8,
                    "num_leaves": 143,
                    "learning_rate": 0.05729454519830571,
                    "min_child_samples": 14,
                    "min_split_gain": 0.3419925921701989,
                    "colsample_bytree": 0.7306103760198094,
                    "subsample": 0.7353233208609915,
                    "reg_alpha": 9.972632608582229,
                    "reg_lambda": 2.9003171689847775,
                    "random_state": 9555
                }
            },
            2020: {
                9555: {
                    "n_estimators": 2795,
                    "max_depth": 13,
                    "num_leaves": 77,
                    "learning_rate": 0.040436242643251914,
                    "min_child_samples": 5,
                    "min_split_gain": 0.9202537123819592,
                    "colsample_bytree": 0.5970006825571672,
                    "subsample": 0.5274929364558207,
                    "reg_alpha": 4.098100839912929,
                    "reg_lambda": 3.0774928504796044,
                    "random_state": 9555
                }
            },
            2021: {
                9555: {
                    "n_estimators": 578,
                    "max_depth": 16,
                    "num_leaves": 98,
                    "learning_rate": 0.07538920752220979,
                    "min_child_samples": 9,
                    "min_split_gain": 0.9409929874602191,
                    "colsample_bytree": 0.6845591179724975,
                    "subsample": 0.5492577235463664,
                    "reg_alpha": 5.879496336078883,
                    "reg_lambda": 0.0074633895899192115,
                    "random_state": 9555
                }
            },
            2022: {
                9555: {
                    "n_estimators": 311,
                    "max_depth": 15,
                    "num_leaves": 144,
                    "learning_rate": 0.030377762414512444,
                    "min_child_samples": 20,
                    "min_split_gain": 0.36605324797503597,
                    "colsample_bytree": 0.11340708229620433,
                    "subsample": 0.9832501559062968,
                    "reg_alpha": 9.635837320241835,
                    "reg_lambda": 3.908516761115513,
                    "random_state": 9555
                }
            }
        },
        "2순": {
            2018: {
                9555: {
                    "n_estimators": 1029,
                    "max_depth": 3,
                    "num_leaves": 80,
                    "learning_rate": 0.15406786315271742,
                    "min_child_samples": 15,
                    "min_split_gain": 0.4861212168709036,
                    "colsample_bytree": 0.57283636961347,
                    "subsample": 0.5945819515189736,
                    "reg_alpha": 1.0391269213941792,
                    "reg_lambda": 5.657672320660499,
                    "random_state": 9555
                }
            },
            2019: {
                9555: {
                    "n_estimators": 406,
                    "max_depth": 4,
                    "num_leaves": 73,
                    "learning_rate": 0.013407529217910894,
                    "min_child_samples": 17,
                    "min_split_gain": 0.09401014184974442,
                    "colsample_bytree": 0.8567492881463071,
                    "subsample": 0.8550020607281649,
                    "reg_alpha": 8.311177855644047,
                    "reg_lambda": 9.44477634264712,
                    "random_state": 9555
                }
            },
            2020: {
                9555: {
                    "n_estimators": 2933,
                    "max_depth": 8,
                    "num_leaves": 74,
                    "learning_rate": 0.10691201155559088,
                    "min_child_samples": 60,
                    "min_split_gain": 0.8931462723028896,
                    "colsample_bytree": 0.8975807994010057,
                    "subsample": 0.8238875622100331,
                    "reg_alpha": 2.9544133525125087,
                    "reg_lambda": 9.282033550968789,
                    "random_state": 9555
                }
            },
            2021: {
                9555: {
                    "n_estimators": 1178,
                    "max_depth": 16,
                    "num_leaves": 66,
                    "learning_rate": 0.14491412123939731,
                    "min_child_samples": 11,
                    "min_split_gain": 0.6450821034073326,
                    "colsample_bytree": 0.4000521069012763,
                    "subsample": 0.889267188267856,
                    "reg_alpha": 7.391804556148638,
                    "reg_lambda": 1.816844318097333,
                    "random_state": 9555
                }
            },
            2022: {
                9555: {
                    "n_estimators": 1707,
                    "max_depth": 6,
                    "num_leaves": 123,
                    "learning_rate": 0.023951359803896503,
                    "min_child_samples": 18,
                    "min_split_gain": 0.82650573866947,
                    "colsample_bytree": 0.25043840335276346,
                    "subsample": 0.7826930825827756,
                    "reg_alpha": 9.716322760665559,
                    "reg_lambda": 1.0917162057593695,
                    "random_state": 9555
                }
            }
        },
        "3순": {
            2018: {
                9555: {
                    "n_estimators": 2443,
                    "max_depth": 15,
                    "num_leaves": 88,
                    "learning_rate": 0.06552277952853693,
                    "min_child_samples": 17,
                    "min_split_gain": 0.7315608801704321,
                    "colsample_bytree": 0.8622464268740975,
                    "subsample": 0.6847414951424815,
                    "reg_alpha": 8.616958059491466,
                    "reg_lambda": 0.0922670516193187,
                    "random_state": 9555
                }
            },
            2019: {
                9555: {
                    "n_estimators": 2282,
                    "max_depth": 15,
                    "num_leaves": 77,
                    "learning_rate": 0.06112249484726315,
                    "min_child_samples": 7,
                    "min_split_gain": 0.6642933033483844,
                    "colsample_bytree": 0.17419335091409993,
                    "subsample": 0.5731722783779221,
                    "reg_alpha": 9.02108088424194,
                    "reg_lambda": 6.231547755122383,
                    "random_state": 9555
                }
            },
            2020: {
                9555: {
                    "n_estimators": 2806,
                    "max_depth": 4,
                    "num_leaves": 48,
                    "learning_rate": 0.09598270750554236,
                    "min_child_samples": 6,
                    "min_split_gain": 0.6410418534242199,
                    "colsample_bytree": 0.2594226440301858,
                    "subsample": 0.7987204980044356,
                    "reg_alpha": 7.955664965399193,
                    "reg_lambda": 3.931759800976117,
                    "random_state": 9555
                }
            },
            2021: {
                9555: {
                    "n_estimators": 2681,
                    "max_depth": 3,
                    "num_leaves": 81,
                    "learning_rate": 0.17555203666818245,
                    "min_child_samples": 12,
                    "min_split_gain": 0.6327277854453546,
                    "colsample_bytree": 0.2573305094906543,
                    "subsample": 0.6773746746309444,
                    "reg_alpha": 4.7161845934063225,
                    "reg_lambda": 1.3419384457868726,
                    "random_state": 9555
                }
            },
            2022: {
                9555: {
                    "n_estimators": 1043,
                    "max_depth": 8,
                    "num_leaves": 66,
                    "learning_rate": 0.01185819055461244,
                    "min_child_samples": 17,
                    "min_split_gain": 0.7098564713207252,
                    "colsample_bytree": 0.49828729853048437,
                    "subsample": 0.6624319036311853,
                    "reg_alpha": 4.093154355471998,
                    "reg_lambda": 4.612637340561466,
                    "random_state": 9555
                }
            }
        }
    },
    "건고추": {
        "1순": {
            2018: {
                9555: {
                    "n_estimators": 1179,
                    "max_depth": 3,
                    "num_leaves": 139,
                    "learning_rate": 0.2791741178734811,
                    "min_child_samples": 12,
                    "min_split_gain": 0.6422683606361395,
                    "colsample_bytree": 0.7847223710409227,
                    "subsample": 0.9528691247986182,
                    "reg_alpha": 6.525365292734226,
                    "reg_lambda": 3.8249263456903377,
                    "random_state": 9555
                }
            },
            2019: {
                9555: {
                    "n_estimators": 2375,
                    "max_depth": 15,
                    "num_leaves": 148,
                    "learning_rate": 0.2225985623348562,
                    "min_child_samples": 17,
                    "min_split_gain": 0.12088100264060425,
                    "colsample_bytree": 0.2587501356563194,
                    "subsample": 0.778892303093098,
                    "reg_alpha": 7.977502372188563,
                    "reg_lambda": 0.6504614137031872,
                    "random_state": 9555
                }
            },
            2020: {
                9555: {
                    "n_estimators": 1070,
                    "max_depth": 11,
                    "num_leaves": 118,
                    "learning_rate": 0.04207484291643758,
                    "min_child_samples": 8,
                    "min_split_gain": 0.708196239860796,
                    "colsample_bytree": 0.9622436157281822,
                    "subsample": 0.9108916317688478,
                    "reg_alpha": 3.945958495652216,
                    "reg_lambda": 0.41031593788559,
                    "random_state": 9555
                }
            },
            2021: {
                9555: {
                    "n_estimators": 2810,
                    "max_depth": 5,
                    "num_leaves": 116,
                    "learning_rate": 0.049820231849461806,
                    "min_child_samples": 11,
                    "min_split_gain": 0.8893443212430698,
                    "colsample_bytree": 0.1760406866494655,
                    "subsample": 0.7358009271287076,
                    "reg_alpha": 7.7274979853494745,
                    "reg_lambda": 3.5672909246581375,
                    "random_state": 9555
                }
            },
            2022: {
                9555: {
                    "n_estimators": 2492,
                    "max_depth": 11,
                    "num_leaves": 82,
                    "learning_rate": 0.24888210580213588,
                    "min_child_samples": 24,
                    "min_split_gain": 0.2802058242932318,
                    "colsample_bytree": 0.7942358438213696,
                    "subsample": 0.5081525959994275,
                    "reg_alpha": 8.047390962594132,
                    "reg_lambda": 5.050707976760463,
                    "random_state": 9555
                }
            }
        },
        "2순": {
            2018: {
                9555: {
                    "n_estimators": 1868,
                    "max_depth": 13,
                    "num_leaves": 32,
                    "learning_rate": 0.2263710244093841,
                    "min_child_samples": 17,
                    "min_split_gain": 0.5206562723263561,
                    "colsample_bytree": 0.19211358590491479,
                    "subsample": 0.9763076811547364,
                    "reg_alpha": 1.6840582458651916,
                    "reg_lambda": 1.1426783809639045,
                    "random_state": 9555
                }
            },
            2019: {
                9555: {
                    "n_estimators": 645,
                    "max_depth": 20,
                    "num_leaves": 110,
                    "learning_rate": 0.011177339221589255,
                    "min_child_samples": 42,
                    "min_split_gain": 0.05404313633911079,
                    "colsample_bytree": 0.7051410656760903,
                    "subsample": 0.5593808708238974,
                    "reg_alpha": 6.148978517077404,
                    "reg_lambda": 2.9177563350668883,
                    "random_state": 9555
                }
            },
            2020: {
                9555: {
                    "n_estimators": 1182,
                    "max_depth": 20,
                    "num_leaves": 91,
                    "learning_rate": 0.24235287796441463,
                    "min_child_samples": 28,
                    "min_split_gain": 0.7056928964814728,
                    "colsample_bytree": 0.5037988797822774,
                    "subsample": 0.8224653065635579,
                    "reg_alpha": 0.879884796692892,
                    "reg_lambda": 0.5267165444537364,
                    "random_state": 9555
                }
            },
            2021: {
                9555: {
                    "n_estimators": 1133,
                    "max_depth": 11,
                    "num_leaves": 105,
                    "learning_rate": 0.16668440214243366,
                    "min_child_samples": 13,
                    "min_split_gain": 0.7321039548225166,
                    "colsample_bytree": 0.45611165287842276,
                    "subsample": 0.7918109751104938,
                    "reg_alpha": 0.11865457809513412,
                    "reg_lambda": 0.43962463227554593,
                    "random_state": 9555
                }
            },
            2022: {
                9555: {
                    "n_estimators": 2971,
                    "max_depth": 12,
                    "num_leaves": 74,
                    "learning_rate": 0.24527879693154736,
                    "min_child_samples": 9,
                    "min_split_gain": 0.8617281108505496,
                    "colsample_bytree": 0.4462464688084934,
                    "subsample": 0.6604070338270844,
                    "reg_alpha": 9.608610840456656,
                    "reg_lambda": 3.471138035503869,
                    "random_state": 9555
                }
            }
        },
        "3순": {
            2018: {
                9555: {
                    "n_estimators": 2858,
                    "max_depth": 19,
                    "num_leaves": 73,
                    "learning_rate": 0.24619723651089284,
                    "min_child_samples": 93,
                    "min_split_gain": 0.014871537364180898,
                    "colsample_bytree": 0.1680777994283182,
                    "subsample": 0.597707335419903,
                    "reg_alpha": 0.6846123166708029,
                    "reg_lambda": 6.718199859725046,
                    "random_state": 9555
                }
            },
            2019: {
                9555: {
                    "n_estimators": 2016,
                    "max_depth": 17,
                    "num_leaves": 97,
                    "learning_rate": 0.1561104745291914,
                    "min_child_samples": 86,
                    "min_split_gain": 0.3981731573272653,
                    "colsample_bytree": 0.5760838235144874,
                    "subsample": 0.8248460343561731,
                    "reg_alpha": 5.690429273793315,
                    "reg_lambda": 0.7294063653482208,
                    "random_state": 9555
                }
            },
            2020: {
                9555: {
                    "n_estimators": 1286,
                    "max_depth": 12,
                    "num_leaves": 78,
                    "learning_rate": 0.1138328621546833,
                    "min_child_samples": 27,
                    "min_split_gain": 0.22539108562432214,
                    "colsample_bytree": 0.48427630654379106,
                    "subsample": 0.7164956615679678,
                    "reg_alpha": 5.405490916454717,
                    "reg_lambda": 7.749476780394635,
                    "random_state": 9555
                }
            },
            2021: {
                9555: {
                    "n_estimators": 1320,
                    "max_depth": 14,
                    "num_leaves": 23,
                    "learning_rate": 0.14946362977006547,
                    "min_child_samples": 19,
                    "min_split_gain": 0.4758047394103208,
                    "colsample_bytree": 0.4174813053727215,
                    "subsample": 0.9507817711255454,
                    "reg_alpha": 6.501881114109343,
                    "reg_lambda": 1.1968249553246302,
                    "random_state": 9555
                }
            },
            2022: {
                9555: {
                    "n_estimators": 1483,
                    "max_depth": 7,
                    "num_leaves": 83,
                    "learning_rate": 0.280346680928165,
                    "min_child_samples": 25,
                    "min_split_gain": 0.1761196976481116,
                    "colsample_bytree": 0.5268387235578421,
                    "subsample": 0.7064770611119318,
                    "reg_alpha": 3.094349755189188,
                    "reg_lambda": 9.929800532954754,
                    "random_state": 9555
                }
            }
        }
    },
    "깐마늘(국산)": {
        "1순": {
            2018: {
                9555: {
                    "n_estimators": 2412,
                    "max_depth": 12,
                    "num_leaves": 144,
                    "learning_rate": 0.051738199743564806,
                    "min_child_samples": 87,
                    "min_split_gain": 0.5549077808586003,
                    "colsample_bytree": 0.5635673585576512,
                    "subsample": 0.6468874576839161,
                    "reg_alpha": 8.639379497573344,
                    "reg_lambda": 2.345305101559169,
                    "random_state": 9555
                }
            },
            2019: {
                9555: {
                    "n_estimators": 2917,
                    "max_depth": 17,
                    "num_leaves": 22,
                    "learning_rate": 0.20876951955488354,
                    "min_child_samples": 100,
                    "min_split_gain": 0.1619910291929495,
                    "colsample_bytree": 0.22955235929090723,
                    "subsample": 0.8608418863058154,
                    "reg_alpha": 6.135492040154421,
                    "reg_lambda": 2.999026349947308,
                    "random_state": 9555
                }
            },
            2020: {
                9555: {
                    "n_estimators": 1499,
                    "max_depth": 16,
                    "num_leaves": 131,
                    "learning_rate": 0.18615109070512934,
                    "min_child_samples": 97,
                    "min_split_gain": 0.664808386392113,
                    "colsample_bytree": 0.4677006025749524,
                    "subsample": 0.6639176609728903,
                    "reg_alpha": 5.55475598683477,
                    "reg_lambda": 4.056210344758999,
                    "random_state": 9555
                }
            },
            2021: {
                9555: {
                    "n_estimators": 1873,
                    "max_depth": 15,
                    "num_leaves": 60,
                    "learning_rate": 0.1597375968413327,
                    "min_child_samples": 20,
                    "min_split_gain": 0.6849639577783231,
                    "colsample_bytree": 0.3211273409734886,
                    "subsample": 0.6573266342694225,
                    "reg_alpha": 4.597400690162742,
                    "reg_lambda": 6.797433078044956,
                    "random_state": 9555
                }
            },
            2022: {
                9555: {
                    "n_estimators": 2352,
                    "max_depth": 19,
                    "num_leaves": 133,
                    "learning_rate": 0.07809979238062893,
                    "min_child_samples": 15,
                    "min_split_gain": 0.177343031926306,
                    "colsample_bytree": 0.23077720642433466,
                    "subsample": 0.8260803923608222,
                    "reg_alpha": 7.352229305199895,
                    "reg_lambda": 6.103309326758655,
                    "random_state": 9555
                }
            }
        },
        "2순": {
            2018: {
                9555: {
                    "n_estimators": 2668,
                    "max_depth": 9,
                    "num_leaves": 114,
                    "learning_rate": 0.2754435730479736,
                    "min_child_samples": 97,
                    "min_split_gain": 0.9795315221979112,
                    "colsample_bytree": 0.8282062651254655,
                    "subsample": 0.9999470333760168,
                    "reg_alpha": 6.408473795265223,
                    "reg_lambda": 7.4359112447605105,
                    "random_state": 9555
                }
            },
            2019: {
                9555: {
                    "n_estimators": 345,
                    "max_depth": 13,
                    "num_leaves": 81,
                    "learning_rate": 0.05101322934441178,
                    "min_child_samples": 76,
                    "min_split_gain": 0.1826865813327978,
                    "colsample_bytree": 0.249337977615463,
                    "subsample": 0.7943934771916998,
                    "reg_alpha": 6.493130961248688,
                    "reg_lambda": 8.005068069203555,
                    "random_state": 9555
                }
            },
            2020: {
                9555: {
                    "n_estimators": 969,
                    "max_depth": 9,
                    "num_leaves": 72,
                    "learning_rate": 0.13259695042653252,
                    "min_child_samples": 86,
                    "min_split_gain": 0.20078273325731089,
                    "colsample_bytree": 0.5408543750827146,
                    "subsample": 0.9312648274413765,
                    "reg_alpha": 6.424581194568171,
                    "reg_lambda": 8.197820977508881,
                    "random_state": 9555
                }
            },
            2021: {
                9555: {
                    "n_estimators": 2264,
                    "max_depth": 15,
                    "num_leaves": 36,
                    "learning_rate": 0.245693835128173,
                    "min_child_samples": 32,
                    "min_split_gain": 0.6072805321766949,
                    "colsample_bytree": 0.7179110955455729,
                    "subsample": 0.5476260813744187,
                    "reg_alpha": 5.04053830824436,
                    "reg_lambda": 9.30282343556047,
                    "random_state": 9555
                }
            },
            2022: {
                9555: {
                    "n_estimators": 712,
                    "max_depth": 5,
                    "num_leaves": 99,
                    "learning_rate": 0.2582316499234327,
                    "min_child_samples": 20,
                    "min_split_gain": 0.3986004803994049,
                    "colsample_bytree": 0.7264959246443293,
                    "subsample": 0.5707017058143332,
                    "reg_alpha": 7.325195718178882,
                    "reg_lambda": 0.8625027313036249,
                    "random_state": 9555
                }
            }
        },
        "3순": {
            2018: {
                9555: {
                    "n_estimators": 311,
                    "max_depth": 14,
                    "num_leaves": 111,
                    "learning_rate": 0.04373903195113291,
                    "min_child_samples": 97,
                    "min_split_gain": 0.5766062585370446,
                    "colsample_bytree": 0.3589283292899721,
                    "subsample": 0.7397968865084923,
                    "reg_alpha": 7.568664331383132,
                    "reg_lambda": 8.977427026179702,
                    "random_state": 9555
                }
            },
            2019: {
                9555: {
                    "n_estimators": 406,
                    "max_depth": 14,
                    "num_leaves": 107,
                    "learning_rate": 0.16921921548118155,
                    "min_child_samples": 83,
                    "min_split_gain": 0.3251410312500371,
                    "colsample_bytree": 0.5843367487964765,
                    "subsample": 0.6056855319096079,
                    "reg_alpha": 9.585190399029216,
                    "reg_lambda": 8.39117427194414,
                    "random_state": 9555
                }
            },
            2020: {
                9555: {
                    "n_estimators": 1360,
                    "max_depth": 14,
                    "num_leaves": 121,
                    "learning_rate": 0.1641259002945558,
                    "min_child_samples": 72,
                    "min_split_gain": 0.6085196348541847,
                    "colsample_bytree": 0.2940810260112274,
                    "subsample": 0.6354166066920182,
                    "reg_alpha": 8.570912340147057,
                    "reg_lambda": 6.014621698340983,
                    "random_state": 9555
                }
            },
            2021: {
                9555: {
                    "n_estimators": 1036,
                    "max_depth": 10,
                    "num_leaves": 135,
                    "learning_rate": 0.2345561587181368,
                    "min_child_samples": 25,
                    "min_split_gain": 0.0928817080197008,
                    "colsample_bytree": 0.7453804482858011,
                    "subsample": 0.8652278669587775,
                    "reg_alpha": 7.581624343384316,
                    "reg_lambda": 3.235249538949735,
                    "random_state": 9555
                }
            },
            2022: {
                9555: {
                    "n_estimators": 1533,
                    "max_depth": 15,
                    "num_leaves": 81,
                    "learning_rate": 0.024536818701966,
                    "min_child_samples": 14,
                    "min_split_gain": 0.4458227308242814,
                    "colsample_bytree": 0.13850328632881698,
                    "subsample": 0.9121163076694421,
                    "reg_alpha": 4.704358258113621,
                    "reg_lambda": 8.44790644694158,
                    "random_state": 9555
                }
            }
        }
    },
    "사과": {
        "1순": {
            2018: {
                9555: {
                    "n_estimators": 1002,
                    "max_depth": 19,
                    "num_leaves": 44,
                    "learning_rate": 0.04181517785159396,
                    "min_child_samples": 32,
                    "min_split_gain": 0.09579457572898581,
                    "colsample_bytree": 0.6374136110953751,
                    "subsample": 0.6381117326027189,
                    "reg_alpha": 2.7233182478521356,
                    "reg_lambda": 8.710026396019996,
                    "random_state": 9555
                }
            },
            2019: {
                9555: {
                    "n_estimators": 1646,
                    "max_depth": 14,
                    "num_leaves": 46,
                    "learning_rate": 0.08270199426660865,
                    "min_child_samples": 34,
                    "min_split_gain": 0.8802379390585989,
                    "colsample_bytree": 0.4188238881946541,
                    "subsample": 0.9324249296635262,
                    "reg_alpha": 0.027604673959835235,
                    "reg_lambda": 4.737838285013699,
                    "random_state": 9555
                }
            },
            2020: {
                9555: {
                    "n_estimators": 1166,
                    "max_depth": 16,
                    "num_leaves": 46,
                    "learning_rate": 0.1916305250725568,
                    "min_child_samples": 62,
                    "min_split_gain": 0.9142417539353008,
                    "colsample_bytree": 0.8370807412791157,
                    "subsample": 0.7624575791013053,
                    "reg_alpha": 2.3350343523040133,
                    "reg_lambda": 3.0630824708721036,
                    "random_state": 9555
                }
            },
            2021: {
                9555: {
                    "n_estimators": 580,
                    "max_depth": 9,
                    "num_leaves": 101,
                    "learning_rate": 0.08734440233418622,
                    "min_child_samples": 19,
                    "min_split_gain": 0.0678887658954576,
                    "colsample_bytree": 0.725050751213517,
                    "subsample": 0.5927671734670648,
                    "reg_alpha": 4.57537540031564,
                    "reg_lambda": 1.8628554908610213,
                    "random_state": 9555
                }
            },
            2022: {
                9555: {
                    "n_estimators": 1673,
                    "max_depth": 6,
                    "num_leaves": 135,
                    "learning_rate": 0.09019687673757211,
                    "min_child_samples": 57,
                    "min_split_gain": 0.5425791312052838,
                    "colsample_bytree": 0.4399662497246768,
                    "subsample": 0.5189555823569251,
                    "reg_alpha": 1.4845797304188584,
                    "reg_lambda": 9.906295793181636,
                    "random_state": 9555
                }
            }
        },
        "2순": {
            2018: {
                9555: {
                    "n_estimators": 311,
                    "max_depth": 6,
                    "num_leaves": 78,
                    "learning_rate": 0.15263434555686767,
                    "min_child_samples": 32,
                    "min_split_gain": 0.35831639393654807,
                    "colsample_bytree": 0.18523900228805235,
                    "subsample": 0.9767490004312004,
                    "reg_alpha": 1.8701581041980164,
                    "reg_lambda": 7.548970916699647,
                    "random_state": 9555
                }
            },
            2019: {
                9555: {
                    "n_estimators": 730,
                    "max_depth": 4,
                    "num_leaves": 60,
                    "learning_rate": 0.13790846697475415,
                    "min_child_samples": 36,
                    "min_split_gain": 0.9259227781562946,
                    "colsample_bytree": 0.18463580616475148,
                    "subsample": 0.816837964123786,
                    "reg_alpha": 9.596213662055833,
                    "reg_lambda": 1.4631029880776598,
                    "random_state": 9555
                }
            },
            2020: {
                9555: {
                    "n_estimators": 1409,
                    "max_depth": 19,
                    "num_leaves": 135,
                    "learning_rate": 0.24128246377591148,
                    "min_child_samples": 34,
                    "min_split_gain": 0.7079118519351066,
                    "colsample_bytree": 0.555988548271754,
                    "subsample": 0.9527772807291388,
                    "reg_alpha": 0.4495220628509138,
                    "reg_lambda": 0.13486151628902332,
                    "random_state": 9555
                }
            },
            2021: {
                9555: {
                    "n_estimators": 1992,
                    "max_depth": 15,
                    "num_leaves": 76,
                    "learning_rate": 0.0415330508169949,
                    "min_child_samples": 16,
                    "min_split_gain": 0.10788972214550668,
                    "colsample_bytree": 0.44552389372579354,
                    "subsample": 0.6621727106805358,
                    "reg_alpha": 0.7275775367719858,
                    "reg_lambda": 1.173141035772275,
                    "random_state": 9555
                }
            },
            2022: {
                9555: {
                    "n_estimators": 2038,
                    "max_depth": 8,
                    "num_leaves": 60,
                    "learning_rate": 0.2661985499843774,
                    "min_child_samples": 16,
                    "min_split_gain": 0.1675602793857141,
                    "colsample_bytree": 0.9108948177896196,
                    "subsample": 0.8039905652431457,
                    "reg_alpha": 5.779011783775561,
                    "reg_lambda": 1.4660261867249853,
                    "random_state": 9555
                }
            }
        },
        "3순": {
            2018: {
                9555: {
                    "n_estimators": 2705,
                    "max_depth": 17,
                    "num_leaves": 50,
                    "learning_rate": 0.2993661116407172,
                    "min_child_samples": 28,
                    "min_split_gain": 0.9917767933860352,
                    "colsample_bytree": 0.4955404664255393,
                    "subsample": 0.7453197805290365,
                    "reg_alpha": 0.11741698338007021,
                    "reg_lambda": 6.234807066075882,
                    "random_state": 9555
                }
            },
            2019: {
                9555: {
                    "n_estimators": 2887,
                    "max_depth": 9,
                    "num_leaves": 95,
                    "learning_rate": 0.12987644610838084,
                    "min_child_samples": 33,
                    "min_split_gain": 0.12409560602071157,
                    "colsample_bytree": 0.7864328478828021,
                    "subsample": 0.8402970083869709,
                    "reg_alpha": 4.733207899686091,
                    "reg_lambda": 7.323399128135808,
                    "random_state": 9555
                }
            },
            2020: {
                9555: {
                    "n_estimators": 688,
                    "max_depth": 10,
                    "num_leaves": 126,
                    "learning_rate": 0.07665576740129226,
                    "min_child_samples": 45,
                    "min_split_gain": 0.18912135336179492,
                    "colsample_bytree": 0.7840602890431824,
                    "subsample": 0.8976227835281275,
                    "reg_alpha": 7.702024549274888,
                    "reg_lambda": 1.8339796364882348,
                    "random_state": 9555
                }
            },
            2021: {
                9555: {
                    "n_estimators": 2036,
                    "max_depth": 15,
                    "num_leaves": 52,
                    "learning_rate": 0.18638755147444716,
                    "min_child_samples": 37,
                    "min_split_gain": 0.3458539274619157,
                    "colsample_bytree": 0.9002654927375032,
                    "subsample": 0.6512588116585721,
                    "reg_alpha": 0.38178824145820683,
                    "reg_lambda": 0.25222494787664473,
                    "random_state": 9555
                }
            },
            2022: {
                9555: {
                    "n_estimators": 1207,
                    "max_depth": 3,
                    "num_leaves": 148,
                    "learning_rate": 0.1604614087295643,
                    "min_child_samples": 57,
                    "min_split_gain": 0.16919903552970292,
                    "colsample_bytree": 0.23843065129641644,
                    "subsample": 0.8563959795300615,
                    "reg_alpha": 8.421939269742074,
                    "reg_lambda": 2.3293739802871034,
                    "random_state": 9555
                }
            }
        }
    },
    "배": {
        "1순": {
            2018: {
                9555: {
                    "n_estimators": 925,
                    "max_depth": 3,
                    "num_leaves": 126,
                    "learning_rate": 0.10331143620182819,
                    "min_child_samples": 9,
                    "min_split_gain": 0.3896204229346259,
                    "colsample_bytree": 0.8714169161878094,
                    "subsample": 0.5734053509094004,
                    "reg_alpha": 9.129544968895832,
                    "reg_lambda": 1.786446749411997,
                    "random_state": 9555
                }
            },
            2019: {
                9555: {
                    "n_estimators": 1096,
                    "max_depth": 7,
                    "num_leaves": 34,
                    "learning_rate": 0.034533144494874685,
                    "min_child_samples": 18,
                    "min_split_gain": 0.6442607035282382,
                    "colsample_bytree": 0.6305831451032428,
                    "subsample": 0.692992894862016,
                    "reg_alpha": 2.3639589540685133,
                    "reg_lambda": 5.642901638895005,
                    "random_state": 9555
                }
            },
            2020: {
                9555: {
                    "n_estimators": 1515,
                    "max_depth": 13,
                    "num_leaves": 111,
                    "learning_rate": 0.03697737133678791,
                    "min_child_samples": 8,
                    "min_split_gain": 0.5816806679753064,
                    "colsample_bytree": 0.7837089820846146,
                    "subsample": 0.9813324786475284,
                    "reg_alpha": 2.5476359551472836,
                    "reg_lambda": 1.7032485037452316,
                    "random_state": 9555
                }
            },
            2021: {
                9555: {
                    "n_estimators": 1534,
                    "max_depth": 19,
                    "num_leaves": 81,
                    "learning_rate": 0.2507118815646876,
                    "min_child_samples": 32,
                    "min_split_gain": 0.7363119289766461,
                    "colsample_bytree": 0.9983314825626065,
                    "subsample": 0.9186172736458019,
                    "reg_alpha": 8.565708462196223,
                    "reg_lambda": 4.726064894350664,
                    "random_state": 9555
                }
            },
            2022: {
                9555: {
                    "n_estimators": 676,
                    "max_depth": 15,
                    "num_leaves": 99,
                    "learning_rate": 0.2256471352940798,
                    "min_child_samples": 58,
                    "min_split_gain": 0.24735889765328312,
                    "colsample_bytree": 0.2207290797941552,
                    "subsample": 0.5552844208087866,
                    "reg_alpha": 3.0039968717055405,
                    "reg_lambda": 0.8690492857302189,
                    "random_state": 9555
                }
            }
        },
        "2순": {
            2018: {
                9555: {
                    "n_estimators": 1211,
                    "max_depth": 14,
                    "num_leaves": 57,
                    "learning_rate": 0.10771378564108106,
                    "min_child_samples": 68,
                    "min_split_gain": 0.5400310110398845,
                    "colsample_bytree": 0.17169498131755073,
                    "subsample": 0.6470828043849277,
                    "reg_alpha": 5.459894161203701,
                    "reg_lambda": 0.5119384559797422,
                    "random_state": 9555
                }
            },
            2019: {
                9555: {
                    "n_estimators": 2789,
                    "max_depth": 19,
                    "num_leaves": 89,
                    "learning_rate": 0.06465343286245032,
                    "min_child_samples": 18,
                    "min_split_gain": 0.9538637947538355,
                    "colsample_bytree": 0.8398998729128053,
                    "subsample": 0.6409753760771839,
                    "reg_alpha": 3.059141448890299,
                    "reg_lambda": 7.700856424799101,
                    "random_state": 9555
                }
            },
            2020: {
                9555: {
                    "n_estimators": 971,
                    "max_depth": 18,
                    "num_leaves": 43,
                    "learning_rate": 0.06720389622927755,
                    "min_child_samples": 18,
                    "min_split_gain": 0.8720572539663813,
                    "colsample_bytree": 0.5154687464109405,
                    "subsample": 0.6636266608116104,
                    "reg_alpha": 5.513787100500515,
                    "reg_lambda": 3.0758336080015796,
                    "random_state": 9555
                }
            },
            2021: {
                9555: {
                    "n_estimators": 391,
                    "max_depth": 15,
                    "num_leaves": 103,
                    "learning_rate": 0.17987373735075324,
                    "min_child_samples": 27,
                    "min_split_gain": 0.41270320455258946,
                    "colsample_bytree": 0.779315791327892,
                    "subsample": 0.9874114724383456,
                    "reg_alpha": 5.996186244072444,
                    "reg_lambda": 6.974244412575038,
                    "random_state": 9555
                }
            },
            2022: {
                9555: {
                    "n_estimators": 2325,
                    "max_depth": 17,
                    "num_leaves": 106,
                    "learning_rate": 0.030851214791884587,
                    "min_child_samples": 58,
                    "min_split_gain": 0.44940289407948913,
                    "colsample_bytree": 0.6018003777728975,
                    "subsample": 0.5215740511900573,
                    "reg_alpha": 1.8569459273377003,
                    "reg_lambda": 3.8884215381583243,
                    "random_state": 9555
                }
            }
        },
        "3순": {
            2018: {
                9555: {
                    "n_estimators": 1115,
                    "max_depth": 7,
                    "num_leaves": 104,
                    "learning_rate": 0.010754904042414217,
                    "min_child_samples": 54,
                    "min_split_gain": 0.7184186137175723,
                    "colsample_bytree": 0.20906348465920277,
                    "subsample": 0.9011473705635188,
                    "reg_alpha": 7.961113393215923,
                    "reg_lambda": 6.51654897901898,
                    "random_state": 9555
                }
            },
            2019: {
                9555: {
                    "n_estimators": 2188,
                    "max_depth": 3,
                    "num_leaves": 127,
                    "learning_rate": 0.04015776768082825,
                    "min_child_samples": 10,
                    "min_split_gain": 0.03275627758531671,
                    "colsample_bytree": 0.9265106162545939,
                    "subsample": 0.5183507246815564,
                    "reg_alpha": 2.3092883904867874,
                    "reg_lambda": 5.129360077152505,
                    "random_state": 9555
                }
            },
            2020: {
                9555: {
                    "n_estimators": 682,
                    "max_depth": 14,
                    "num_leaves": 111,
                    "learning_rate": 0.03026003580325113,
                    "min_child_samples": 5,
                    "min_split_gain": 0.8986274325868349,
                    "colsample_bytree": 0.4756564976216649,
                    "subsample": 0.9166958855576832,
                    "reg_alpha": 2.0117153122729166,
                    "reg_lambda": 9.620941671982381,
                    "random_state": 9555
                }
            },
            2021: {
                9555: {
                    "n_estimators": 1717,
                    "max_depth": 3,
                    "num_leaves": 57,
                    "learning_rate": 0.2492813103135674,
                    "min_child_samples": 14,
                    "min_split_gain": 0.7699647712048994,
                    "colsample_bytree": 0.6591316033954304,
                    "subsample": 0.8364798031935856,
                    "reg_alpha": 2.543665580789761,
                    "reg_lambda": 4.090831928468542,
                    "random_state": 9555
                }
            },
            2022: {
                9555: {
                    "n_estimators": 1012,
                    "max_depth": 7,
                    "num_leaves": 40,
                    "learning_rate": 0.2747415772827873,
                    "min_child_samples": 64,
                    "min_split_gain": 0.853701364209853,
                    "colsample_bytree": 0.9775808993916333,
                    "subsample": 0.5303218837056798,
                    "reg_alpha": 8.920710510549226,
                    "reg_lambda": 5.2960104554013565,
                    "random_state": 9555
                }
            }
        }
    },
    "상추": {
        "1순": {
            2018: {
                9555: {
                    "n_estimators": 2055,
                    "max_depth": 7,
                    "num_leaves": 115,
                    "learning_rate": 0.02257292462612947,
                    "min_child_samples": 28,
                    "min_split_gain": 0.24604963064039098,
                    "colsample_bytree": 0.7808759122942681,
                    "subsample": 0.8627445880068787,
                    "reg_alpha": 5.175344371253922,
                    "reg_lambda": 7.825098893535551,
                    "random_state": 9555
                }
            },
            2019: {
                9555: {
                    "n_estimators": 2630,
                    "max_depth": 17,
                    "num_leaves": 69,
                    "learning_rate": 0.1844222138912305,
                    "min_child_samples": 26,
                    "min_split_gain": 0.11055768094108434,
                    "colsample_bytree": 0.44609109076043285,
                    "subsample": 0.6779777529343881,
                    "reg_alpha": 8.876087717502198,
                    "reg_lambda": 0.5417969554685733,
                    "random_state": 9555
                }
            },
            2020: {
                9555: {
                    "n_estimators": 1103,
                    "max_depth": 9,
                    "num_leaves": 145,
                    "learning_rate": 0.08782387076224997,
                    "min_child_samples": 23,
                    "min_split_gain": 0.7436836147733644,
                    "colsample_bytree": 0.20350035360534624,
                    "subsample": 0.7457175910332197,
                    "reg_alpha": 8.60538923896946,
                    "reg_lambda": 9.054827769521403,
                    "random_state": 9555
                }
            },
            2021: {
                9555: {
                    "n_estimators": 2050,
                    "max_depth": 17,
                    "num_leaves": 55,
                    "learning_rate": 0.06521501727040814,
                    "min_child_samples": 22,
                    "min_split_gain": 0.04919782556895219,
                    "colsample_bytree": 0.7554300797438529,
                    "subsample": 0.5757910432684057,
                    "reg_alpha": 1.7516247384496748,
                    "reg_lambda": 8.205117629604064,
                    "random_state": 9555
                }
            },
            2022: {
                9555: {
                    "n_estimators": 1672,
                    "max_depth": 19,
                    "num_leaves": 90,
                    "learning_rate": 0.15746899585755017,
                    "min_child_samples": 5,
                    "min_split_gain": 0.545273042608847,
                    "colsample_bytree": 0.5742214745322166,
                    "subsample": 0.5580487507223335,
                    "reg_alpha": 7.13326174853631,
                    "reg_lambda": 6.0498498787103046,
                    "random_state": 9555
                }
            }
        },
        "2순": {
            2018: {
                9555: {
                    "n_estimators": 2937,
                    "max_depth": 13,
                    "num_leaves": 37,
                    "learning_rate": 0.05110137377002981,
                    "min_child_samples": 9,
                    "min_split_gain": 0.27722204099928294,
                    "colsample_bytree": 0.2480476554949681,
                    "subsample": 0.6357059818442995,
                    "reg_alpha": 8.109786276659982,
                    "reg_lambda": 2.356126705239057,
                    "random_state": 9555
                }
            },
            2019: {
                9555: {
                    "n_estimators": 2543,
                    "max_depth": 14,
                    "num_leaves": 117,
                    "learning_rate": 0.1875385634811214,
                    "min_child_samples": 8,
                    "min_split_gain": 0.5382088680023202,
                    "colsample_bytree": 0.9515458887526554,
                    "subsample": 0.8806316997003271,
                    "reg_alpha": 5.372497332823012,
                    "reg_lambda": 8.93217821158856,
                    "random_state": 9555
                }
            },
            2020: {
                9555: {
                    "n_estimators": 514,
                    "max_depth": 17,
                    "num_leaves": 59,
                    "learning_rate": 0.02317819544580166,
                    "min_child_samples": 35,
                    "min_split_gain": 0.19032988856473096,
                    "colsample_bytree": 0.48633136162083757,
                    "subsample": 0.7907121070674655,
                    "reg_alpha": 9.018907321569372,
                    "reg_lambda": 5.863981448820709,
                    "random_state": 9555
                }
            },
            2021: {
                9555: {
                    "n_estimators": 406,
                    "max_depth": 4,
                    "num_leaves": 20,
                    "learning_rate": 0.29095033944024173,
                    "min_child_samples": 5,
                    "min_split_gain": 0.6273974312916152,
                    "colsample_bytree": 0.10913010078548785,
                    "subsample": 0.9983449974056855,
                    "reg_alpha": 0.08702215715429312,
                    "reg_lambda": 0.8205346942339817,
                    "random_state": 9555
                }
            },
            2022: {
                9555: {
                    "n_estimators": 704,
                    "max_depth": 9,
                    "num_leaves": 102,
                    "learning_rate": 0.047423717568557386,
                    "min_child_samples": 30,
                    "min_split_gain": 0.029420622207823675,
                    "colsample_bytree": 0.1265419803872201,
                    "subsample": 0.7770478175801405,
                    "reg_alpha": 7.993021223444785,
                    "reg_lambda": 1.2041599967532557,
                    "random_state": 9555
                }
            }
        },
        "3순": {
            2018: {
                9555: {
                    "n_estimators": 2018,
                    "max_depth": 9,
                    "num_leaves": 85,
                    "learning_rate": 0.07874294492159659,
                    "min_child_samples": 13,
                    "min_split_gain": 0.3188376952878418,
                    "colsample_bytree": 0.4104769482912156,
                    "subsample": 0.8680894012614094,
                    "reg_alpha": 2.0748458297504797,
                    "reg_lambda": 0.027443137064534773,
                    "random_state": 9555
                }
            },
            2019: {
                9555: {
                    "n_estimators": 2654,
                    "max_depth": 6,
                    "num_leaves": 46,
                    "learning_rate": 0.2306612300879917,
                    "min_child_samples": 13,
                    "min_split_gain": 0.11918075557878005,
                    "colsample_bytree": 0.5576053680503279,
                    "subsample": 0.9539753288601585,
                    "reg_alpha": 4.454400973097954,
                    "reg_lambda": 5.232402746153494,
                    "random_state": 9555
                }
            },
            2020: {
                9555: {
                    "n_estimators": 940,
                    "max_depth": 10,
                    "num_leaves": 22,
                    "learning_rate": 0.010223104620230461,
                    "min_child_samples": 28,
                    "min_split_gain": 0.18461467694293848,
                    "colsample_bytree": 0.8641823758503526,
                    "subsample": 0.5342957281241,
                    "reg_alpha": 6.5026513996309445,
                    "reg_lambda": 7.750169185383366,
                    "random_state": 9555
                }
            },
            2021: {
                9555: {
                    "n_estimators": 1112,
                    "max_depth": 9,
                    "num_leaves": 36,
                    "learning_rate": 0.06253874045607044,
                    "min_child_samples": 11,
                    "min_split_gain": 0.004598560611885544,
                    "colsample_bytree": 0.5170656067282268,
                    "subsample": 0.5020614866137145,
                    "reg_alpha": 4.631056282248625,
                    "reg_lambda": 5.922590800578918,
                    "random_state": 9555
                }
            },
            2022: {
                9555: {
                    "n_estimators": 1437,
                    "max_depth": 7,
                    "num_leaves": 20,
                    "learning_rate": 0.25534429979754414,
                    "min_child_samples": 33,
                    "min_split_gain": 0.9881945218873233,
                    "colsample_bytree": 0.8647691655798814,
                    "subsample": 0.6833466621406465,
                    "reg_alpha": 5.792721327927501,
                    "reg_lambda": 6.482736881840644,
                    "random_state": 9555
                }
            }
        }
    }
}

for c in category:
    print(f'\n\n\n============================= {c} =============================')
    train = pd.read_csv(f"train_{c}.csv", encoding='cp949')
    test = pd.read_csv(f"test_{c}.csv", encoding='cp949')
    
    dl_pred = pd.read_csv('dl_model1.csv')
    dl_pred_c = dl_pred[['시점'] + [c]]

    train_origin = train.copy()
    test_val = test.copy()
    
    train, test = common_FE(train, test, c)
    y_preds = {}
    models = {}
    train.drop(['종가'], axis=1, inplace=True)
    test.drop(['종가'], axis=1, inplace=True)

    train_x = train.drop(['1순', '2순', '3순'], axis=1)
    train_y = train[['1순', '2순', '3순']].copy()

    test_original = test.drop(['1순', '2순', '3순'], axis=1)
    test_original.drop(['연도'],axis=1,inplace=True)

    final_models = {}
    y_preds = {}
    seed_preds = []


    seeds = [9555] 
    unique_years = [2018,2019,2020,2021,2022]

    for i in range(1, 4):  # For 1순, 2순, and 3순
        print(f'--------------------- {i}순 Optuna Optimization ---------------------')
        year_preds = []

        for year in unique_years:
            print(f'--------------------- Year {year} Excluded ---------------------')
            train_excluded = train[train['연도'] != year].copy()
            train_excluded.drop(['연도'], axis=1, inplace=True)
            train_excluded = train_excluded.reset_index(drop=True)

            test_x = test_original.copy()

            if i == 1:
                X = train_excluded[train_excluded['1순'] > 0].drop(columns=['1순','2순','3순'])
                y = train_excluded[train_excluded['1순'] > 0]['1순']

            elif i == 2:
                X = train_excluded[train_excluded['2순'] > 0].drop(columns=['1순','2순','3순'])
                y = train_excluded[train_excluded['2순'] > 0]['2순']
                X['pred_1순'] = train_y['1순']
                
                dl_pred_c_1 = dl_pred[dl_pred['시점'].str.contains('\+1순')]
                dl_pred_c_1 = dl_pred_c_1.reset_index(drop=True)
                test_x['pred_1순'] = y_preds['y_pred_1'] * 0.0 + dl_pred_c_1[c] * 1.0
                
            elif i == 3:
                X = train_excluded[train_excluded['3순'] > 0].drop(columns=['1순','2순','3순'])
                y = train_excluded[train_excluded['3순'] > 0]['3순']
                X['pred_1순'] = train_y['1순']
                X['pred_2순'] = train_y['2순']

                dl_pred_c_1 = dl_pred[dl_pred['시점'].str.contains('\+1순')]
                dl_pred_c_1 = dl_pred_c_1.reset_index(drop=True)
                test_x['pred_1순'] = y_preds['y_pred_1'] * 0.0 + dl_pred_c_1[c] * 1.0

                dl_pred_c_2 = dl_pred[dl_pred['시점'].str.contains('\+2순')]
                dl_pred_c_2 = dl_pred_c_2.reset_index(drop=True)
                test_x['pred_2순'] = y_preds['y_pred_2'] * 0.25 + dl_pred_c_2[c] * 0.75

            seed_test_preds = []

            for seed in seeds:
                print(f'--------------------- Seed {seed} ---------------------')
                best_params = best_params_all[c][f'{i}순'][year][seed]
                best_params.pop('random_state')

                best_params['random_state'] = seed
                best_model = lgb.LGBMRegressor(**best_params)
                best_model.fit(X, y)

                seed_test_preds.append(best_model.predict(test_x))

            year_preds.append(np.mean(seed_test_preds, axis=0))

        final_test_pred = np.mean(year_preds, axis=0)
        y_preds[f'y_pred_{i}'] = np.mean(year_preds, axis=0)

    if c in group1:
        sub_test_1.loc[sub_test_1['품목(품종)명'] == c, '1순'] = y_preds['y_pred_1']
        sub_test_1.loc[sub_test_1['품목(품종)명'] == c, '2순'] = y_preds['y_pred_2']
        sub_test_1.loc[sub_test_1['품목(품종)명'] == c, '3순'] = y_preds['y_pred_3']
    elif c in group2:
        sub_test_2.loc[sub_test_2['품목명'] == c, '1순'] = y_preds['y_pred_1']
        sub_test_2.loc[sub_test_2['품목명'] == c, '2순'] = y_preds['y_pred_2']
        sub_test_2.loc[sub_test_2['품목명'] == c, '3순'] = y_preds['y_pred_3']




submission_path = 'sample_submission.csv'

final_test_data = sub_test_1.copy()
submission_data = pd.read_csv(submission_path, encoding='utf-8')
submission_data2 = submission_data.copy()
submission_data3 = submission_data.copy()
submission_data = submission_data[['시점','배추', '무', '양파', '감자 수미', '대파(일반)']]

test_columns = ['시점', '품목(품종)명', '1순', '2순', '3순']
final_test_filtered = final_test_data[test_columns]

pivot_test_data = final_test_filtered.pivot(index='시점', columns='품목(품종)명', values=['1순', '2순', '3순'])

for step in ['1순', '2순', '3순']:
    for product in submission_data.columns[1:]: 
        col_name = f'{step}'
        submission_data.loc[submission_data['시점'].str.contains(step, regex=False), product] = pivot_test_data[col_name, product].values

final_test_data = sub_test_2.copy()

submission_data2 = submission_data2[['시점','건고추', '깐마늘(국산)', '상추', '사과', '배']]
test_columns = ['시점', '품목명', '1순', '2순', '3순']
final_test_filtered = final_test_data[test_columns]

pivot_test_data = final_test_filtered.pivot(index='시점', columns='품목명', values=['1순', '2순', '3순'])

for step in ['1순', '2순', '3순']:
    for product in submission_data2.columns[1:]: 
        col_name = f'{step}'
        submission_data2.loc[submission_data2['시점'].str.contains(step, regex=False), product] = pivot_test_data[col_name, product].values

submission_data3['배추'] = submission_data['배추']
submission_data3['무'] = submission_data['무']
submission_data3['양파'] = submission_data['양파']
submission_data3['감자 수미'] = submission_data['감자 수미']
submission_data3['대파(일반)'] = submission_data['대파(일반)']

submission_data3['건고추'] = submission_data2['건고추']
submission_data3['깐마늘(국산)'] = submission_data2['깐마늘(국산)']
submission_data3['상추'] = submission_data2['상추']
submission_data3['사과'] = submission_data2['사과']
submission_data3['배'] = submission_data2['배']
submission_data3.to_csv('시계열회귀.csv',index=False,encoding='utf-8')
