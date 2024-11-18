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
import numpy as np
import random

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
                724: {
                    "n_estimators": 2225,
                    "max_depth": 9,
                    "num_leaves": 48,
                    "learning_rate": 0.06006103059458847,
                    "min_child_samples": 13,
                    "min_split_gain": 0.47920358126530943,
                    "colsample_bytree": 0.3884541806177846,
                    "subsample": 0.6986892021704617,
                    "reg_alpha": 7.647741536357965,
                    "reg_lambda": 5.758325127037205,
                    "random_state": 724
                }
            },
            2019: {
                724: {
                    "n_estimators": 551,
                    "max_depth": 7,
                    "num_leaves": 141,
                    "learning_rate": 0.1733602105862647,
                    "min_child_samples": 9,
                    "min_split_gain": 0.31698103850567255,
                    "colsample_bytree": 0.1904751279366612,
                    "subsample": 0.6200518126491591,
                    "reg_alpha": 8.748284745485655,
                    "reg_lambda": 3.5280299032895064,
                    "random_state": 724
                }
            },
            2020: {
                724: {
                    "n_estimators": 2741,
                    "max_depth": 12,
                    "num_leaves": 31,
                    "learning_rate": 0.011134958471093259,
                    "min_child_samples": 5,
                    "min_split_gain": 0.688204797056913,
                    "colsample_bytree": 0.18047985692720883,
                    "subsample": 0.5096786788006382,
                    "reg_alpha": 7.8771878938409925,
                    "reg_lambda": 4.430884361071892,
                    "random_state": 724
                }
            },
            2021: {
                724: {
                    "n_estimators": 2728,
                    "max_depth": 4,
                    "num_leaves": 133,
                    "learning_rate": 0.07406454720261653,
                    "min_child_samples": 5,
                    "min_split_gain": 0.5944279060497302,
                    "colsample_bytree": 0.4971359750163783,
                    "subsample": 0.6845195671247593,
                    "reg_alpha": 1.8844910928285148,
                    "reg_lambda": 0.8440455457420557,
                    "random_state": 724
                }
            },
            2022: {
                724: {
                    "n_estimators": 499,
                    "max_depth": 19,
                    "num_leaves": 26,
                    "learning_rate": 0.02997293575817412,
                    "min_child_samples": 5,
                    "min_split_gain": 0.0570926120131916,
                    "colsample_bytree": 0.6004432303770189,
                    "subsample": 0.969270391414826,
                    "reg_alpha": 6.223429006150281,
                    "reg_lambda": 8.667537476163496,
                    "random_state": 724
                }
            }
        },
        "2순": {
            2018: {
                724: {
                    "n_estimators": 399,
                    "max_depth": 3,
                    "num_leaves": 30,
                    "learning_rate": 0.29974973308619873,
                    "min_child_samples": 5,
                    "min_split_gain": 0.9992645896841952,
                    "colsample_bytree": 0.37898818509821675,
                    "subsample": 0.9847144425099412,
                    "reg_alpha": 9.57487063435245,
                    "reg_lambda": 0.5018004753380012,
                    "random_state": 724
                }
            },
            2019: {
                724: {
                    "n_estimators": 843,
                    "max_depth": 12,
                    "num_leaves": 110,
                    "learning_rate": 0.22628416527267817,
                    "min_child_samples": 6,
                    "min_split_gain": 0.49698785620382707,
                    "colsample_bytree": 0.7625611304551326,
                    "subsample": 0.7632648690310435,
                    "reg_alpha": 1.796634465544574,
                    "reg_lambda": 2.791931068028682,
                    "random_state": 724
                }
            },
            2020: {
                724: {
                    "n_estimators": 694,
                    "max_depth": 17,
                    "num_leaves": 53,
                    "learning_rate": 0.22540908101870016,
                    "min_child_samples": 23,
                    "min_split_gain": 0.7838610923723369,
                    "colsample_bytree": 0.19921077014185332,
                    "subsample": 0.5327591140027388,
                    "reg_alpha": 3.2226713070365705,
                    "reg_lambda": 6.298966075200195,
                    "random_state": 724
                }
            },
            2021: {
                724: {
                    "n_estimators": 2906,
                    "max_depth": 6,
                    "num_leaves": 43,
                    "learning_rate": 0.054958387544884364,
                    "min_child_samples": 20,
                    "min_split_gain": 0.9222283679911529,
                    "colsample_bytree": 0.10431134285683052,
                    "subsample": 0.9008282622579353,
                    "reg_alpha": 5.43193370506207,
                    "reg_lambda": 2.994416065081806,
                    "random_state": 724
                }
            },
            2022: {
                724: {
                    "n_estimators": 2039,
                    "max_depth": 9,
                    "num_leaves": 20,
                    "learning_rate": 0.08568618015860296,
                    "min_child_samples": 5,
                    "min_split_gain": 0.19120782851166404,
                    "colsample_bytree": 0.16679390155845583,
                    "subsample": 0.8824183007585475,
                    "reg_alpha": 0.5088306318476391,
                    "reg_lambda": 6.561616392669386,
                    "random_state": 724
                }
            }
        },
        "3순": {
            2018: {
                724: {
                    "n_estimators": 1491,
                    "max_depth": 19,
                    "num_leaves": 44,
                    "learning_rate": 0.056339095767300174,
                    "min_child_samples": 9,
                    "min_split_gain": 0.24983834396753649,
                    "colsample_bytree": 0.7323869882302962,
                    "subsample": 0.9300863206913325,
                    "reg_alpha": 7.226220765883598,
                    "reg_lambda": 9.780490018413499,
                    "random_state": 724
                }
            },
            2019: {
                724: {
                    "n_estimators": 827,
                    "max_depth": 11,
                    "num_leaves": 71,
                    "learning_rate": 0.15064358807617895,
                    "min_child_samples": 12,
                    "min_split_gain": 0.6370461205280068,
                    "colsample_bytree": 0.8732022868683328,
                    "subsample": 0.5483325768293688,
                    "reg_alpha": 2.072138885058622,
                    "reg_lambda": 4.410343970985439,
                    "random_state": 724
                }
            },
            2020: {
                724: {
                    "n_estimators": 2662,
                    "max_depth": 14,
                    "num_leaves": 70,
                    "learning_rate": 0.1872014965386287,
                    "min_child_samples": 5,
                    "min_split_gain": 0.5486660301795576,
                    "colsample_bytree": 0.7945854803229951,
                    "subsample": 0.5045503432722133,
                    "reg_alpha": 3.244270454280782,
                    "reg_lambda": 3.1225684098394604,
                    "random_state": 724
                }
            },
            2021: {
                724: {
                    "n_estimators": 1722,
                    "max_depth": 6,
                    "num_leaves": 66,
                    "learning_rate": 0.24983175309655267,
                    "min_child_samples": 5,
                    "min_split_gain": 0.5249567895327062,
                    "colsample_bytree": 0.8841846421135248,
                    "subsample": 0.6869457115542796,
                    "reg_alpha": 5.184930454249275,
                    "reg_lambda": 6.7775888769536206,
                    "random_state": 724
                }
            },
            2022: {
                724: {
                    "n_estimators": 1425,
                    "max_depth": 16,
                    "num_leaves": 112,
                    "learning_rate": 0.266688766703836,
                    "min_child_samples": 16,
                    "min_split_gain": 0.7832821946249504,
                    "colsample_bytree": 0.7102331851305724,
                    "subsample": 0.9190971448773306,
                    "reg_alpha": 6.026874389693805,
                    "reg_lambda": 8.515628642919744,
                    "random_state": 724
                }
            }
        }
    },
    "무": {
        "1순": {
            2018: {
                724: {
                    "n_estimators": 2302,
                    "max_depth": 3,
                    "num_leaves": 37,
                    "learning_rate": 0.11953226571682762,
                    "min_child_samples": 41,
                    "min_split_gain": 0.4068577026979175,
                    "colsample_bytree": 0.17349328811927223,
                    "subsample": 0.7652029513697249,
                    "reg_alpha": 5.692109348947704,
                    "reg_lambda": 2.953940381047689,
                    "random_state": 724
                }
            },
            2019: {
                724: {
                    "n_estimators": 576,
                    "max_depth": 4,
                    "num_leaves": 39,
                    "learning_rate": 0.08860525143510696,
                    "min_child_samples": 37,
                    "min_split_gain": 0.10196165302588639,
                    "colsample_bytree": 0.17647131523259502,
                    "subsample": 0.8833253030307731,
                    "reg_alpha": 0.8879195655415761,
                    "reg_lambda": 5.556497539772282,
                    "random_state": 724
                }
            },
            2020: {
                724: {
                    "n_estimators": 333,
                    "max_depth": 4,
                    "num_leaves": 135,
                    "learning_rate": 0.05770914517155852,
                    "min_child_samples": 22,
                    "min_split_gain": 0.42162979453546157,
                    "colsample_bytree": 0.4218862677291481,
                    "subsample": 0.87387768085582,
                    "reg_alpha": 4.360086590112798,
                    "reg_lambda": 7.846590044841202,
                    "random_state": 724
                }
            },
            2021: {
                724: {
                    "n_estimators": 2992,
                    "max_depth": 17,
                    "num_leaves": 100,
                    "learning_rate": 0.03311983540832111,
                    "min_child_samples": 5,
                    "min_split_gain": 0.41272368876264903,
                    "colsample_bytree": 0.14001294967246192,
                    "subsample": 0.5005979322528524,
                    "reg_alpha": 5.104853725891661,
                    "reg_lambda": 5.949718532724021,
                    "random_state": 724
                }
            },
            2022: {
                724: {
                    "n_estimators": 1567,
                    "max_depth": 4,
                    "num_leaves": 57,
                    "learning_rate": 0.17121759694326646,
                    "min_child_samples": 38,
                    "min_split_gain": 0.6086385703963131,
                    "colsample_bytree": 0.7357339982292893,
                    "subsample": 0.5617225018076777,
                    "reg_alpha": 0.8742127698739185,
                    "reg_lambda": 4.070350572531045,
                    "random_state": 724
                }
            }
        },
        "2순": {
            2018: {
                724: {
                    "n_estimators": 2405,
                    "max_depth": 4,
                    "num_leaves": 144,
                    "learning_rate": 0.09682467198224506,
                    "min_child_samples": 49,
                    "min_split_gain": 0.19642586847630544,
                    "colsample_bytree": 0.16129315937689295,
                    "subsample": 0.6360204049854005,
                    "reg_alpha": 9.614703520175407,
                    "reg_lambda": 9.755948399135296,
                    "random_state": 724
                }
            },
            2019: {
                724: {
                    "n_estimators": 1037,
                    "max_depth": 20,
                    "num_leaves": 101,
                    "learning_rate": 0.18737714001489772,
                    "min_child_samples": 6,
                    "min_split_gain": 0.8737719289399,
                    "colsample_bytree": 0.2604121879412502,
                    "subsample": 0.8114634835151271,
                    "reg_alpha": 3.657446887469138,
                    "reg_lambda": 7.245591346874643,
                    "random_state": 724
                }
            },
            2020: {
                724: {
                    "n_estimators": 2434,
                    "max_depth": 3,
                    "num_leaves": 96,
                    "learning_rate": 0.12502971360238213,
                    "min_child_samples": 25,
                    "min_split_gain": 0.9990720665714503,
                    "colsample_bytree": 0.7763355676879943,
                    "subsample": 0.6052898994224426,
                    "reg_alpha": 3.6327882485078318,
                    "reg_lambda": 3.1279721337671598,
                    "random_state": 724
                }
            },
            2021: {
                724: {
                    "n_estimators": 302,
                    "max_depth": 20,
                    "num_leaves": 70,
                    "learning_rate": 0.015739105949432425,
                    "min_child_samples": 39,
                    "min_split_gain": 0.0213577929350513,
                    "colsample_bytree": 0.11606397267983179,
                    "subsample": 0.5713748105402806,
                    "reg_alpha": 4.104494057846074,
                    "reg_lambda": 0.06417518877834993,
                    "random_state": 724
                }
            },
            2022: {
                724: {
                    "n_estimators": 865,
                    "max_depth": 10,
                    "num_leaves": 142,
                    "learning_rate": 0.2644103216493792,
                    "min_child_samples": 41,
                    "min_split_gain": 0.4199968943595135,
                    "colsample_bytree": 0.24345757245154395,
                    "subsample": 0.610639320946675,
                    "reg_alpha": 9.341662097734208,
                    "reg_lambda": 2.0033533885904165,
                    "random_state": 724
                }
            }
        },
        "3순": {
            2018: {
                724: {
                    "n_estimators": 767,
                    "max_depth": 17,
                    "num_leaves": 63,
                    "learning_rate": 0.08027133306765091,
                    "min_child_samples": 22,
                    "min_split_gain": 0.20183801473879326,
                    "colsample_bytree": 0.4383019420137766,
                    "subsample": 0.6654491240148961,
                    "reg_alpha": 3.0146241942746768,
                    "reg_lambda": 2.4939702216885684,
                    "random_state": 724
                }
            },
            2019: {
                724: {
                    "n_estimators": 1309,
                    "max_depth": 14,
                    "num_leaves": 127,
                    "learning_rate": 0.12389060317215667,
                    "min_child_samples": 42,
                    "min_split_gain": 0.8150291667640295,
                    "colsample_bytree": 0.5628477239631908,
                    "subsample": 0.9370489072996993,
                    "reg_alpha": 8.279295159921721,
                    "reg_lambda": 4.3165800703245205,
                    "random_state": 724
                }
            },
            2020: {
                724: {
                    "n_estimators": 365,
                    "max_depth": 19,
                    "num_leaves": 103,
                    "learning_rate": 0.10009759785764792,
                    "min_child_samples": 22,
                    "min_split_gain": 0.9971195436539223,
                    "colsample_bytree": 0.39239072294045224,
                    "subsample": 0.7821439430034088,
                    "reg_alpha": 5.562614578017689,
                    "reg_lambda": 6.819315598967757,
                    "random_state": 724
                }
            },
            2021: {
                724: {
                    "n_estimators": 729,
                    "max_depth": 17,
                    "num_leaves": 138,
                    "learning_rate": 0.061371836630880264,
                    "min_child_samples": 48,
                    "min_split_gain": 0.5672658334371347,
                    "colsample_bytree": 0.23502279483924468,
                    "subsample": 0.770414948927453,
                    "reg_alpha": 6.864804319438608,
                    "reg_lambda": 5.532383247253582,
                    "random_state": 724
                }
            },
            2022: {
                724: {
                    "n_estimators": 677,
                    "max_depth": 20,
                    "num_leaves": 49,
                    "learning_rate": 0.14926600607835136,
                    "min_child_samples": 23,
                    "min_split_gain": 0.5156419700651494,
                    "colsample_bytree": 0.20161017264602807,
                    "subsample": 0.6038442488541702,
                    "reg_alpha": 3.371799488766061,
                    "reg_lambda": 3.0955923930141043,
                    "random_state": 724
                }
            }
        }
    },
    "양파": {
        "1순": {
            2018: {
                724: {
                    "n_estimators": 1424,
                    "max_depth": 6,
                    "num_leaves": 136,
                    "learning_rate": 0.2517954562427576,
                    "min_child_samples": 56,
                    "min_split_gain": 0.6844783604258665,
                    "colsample_bytree": 0.6284547170513501,
                    "subsample": 0.5151000936153665,
                    "reg_alpha": 6.442856043592126,
                    "reg_lambda": 8.742379970492903,
                    "random_state": 724
                }
            },
            2019: {
                724: {
                    "n_estimators": 2461,
                    "max_depth": 3,
                    "num_leaves": 41,
                    "learning_rate": 0.20154399283193253,
                    "min_child_samples": 5,
                    "min_split_gain": 0.629376855467705,
                    "colsample_bytree": 0.49433052297879765,
                    "subsample": 0.6387926427095862,
                    "reg_alpha": 1.4921321586393845,
                    "reg_lambda": 3.173355230907524,
                    "random_state": 724
                }
            },
            2020: {
                724: {
                    "n_estimators": 948,
                    "max_depth": 20,
                    "num_leaves": 131,
                    "learning_rate": 0.2184045963649994,
                    "min_child_samples": 7,
                    "min_split_gain": 0.6154075030279789,
                    "colsample_bytree": 0.12780500325646676,
                    "subsample": 0.9969393565026553,
                    "reg_alpha": 0.2815186121053477,
                    "reg_lambda": 0.03558480070135772,
                    "random_state": 724
                }
            },
            2021: {
                724: {
                    "n_estimators": 2492,
                    "max_depth": 10,
                    "num_leaves": 94,
                    "learning_rate": 0.15242746667419735,
                    "min_child_samples": 66,
                    "min_split_gain": 0.6641539989934571,
                    "colsample_bytree": 0.22653205944475,
                    "subsample": 0.608979577571426,
                    "reg_alpha": 9.007735882582356,
                    "reg_lambda": 7.13107349423893,
                    "random_state": 724
                }
            },
            2022: {
                724: {
                    "n_estimators": 1353,
                    "max_depth": 3,
                    "num_leaves": 60,
                    "learning_rate": 0.05510086044781063,
                    "min_child_samples": 27,
                    "min_split_gain": 0.6121759427392023,
                    "colsample_bytree": 0.3212990694446953,
                    "subsample": 0.7670383889769552,
                    "reg_alpha": 5.652490164820071,
                    "reg_lambda": 5.759416161350885,
                    "random_state": 724
                }
            }
        },
        "2순": {
            2018: {
                724: {
                    "n_estimators": 839,
                    "max_depth": 16,
                    "num_leaves": 125,
                    "learning_rate": 0.010913861628741873,
                    "min_child_samples": 55,
                    "min_split_gain": 0.5954751177089487,
                    "colsample_bytree": 0.8088801867144289,
                    "subsample": 0.7482696437933674,
                    "reg_alpha": 2.5736199149501497,
                    "reg_lambda": 1.7268893613605933,
                    "random_state": 724
                }
            },
            2019: {
                724: {
                    "n_estimators": 513,
                    "max_depth": 20,
                    "num_leaves": 130,
                    "learning_rate": 0.03835724022247204,
                    "min_child_samples": 47,
                    "min_split_gain": 0.7439287038546514,
                    "colsample_bytree": 0.9906665755437484,
                    "subsample": 0.6234374825828615,
                    "reg_alpha": 0.4035930301759906,
                    "reg_lambda": 2.0615091263001273,
                    "random_state": 724
                }
            },
            2020: {
                724: {
                    "n_estimators": 712,
                    "max_depth": 15,
                    "num_leaves": 58,
                    "learning_rate": 0.21650807633387747,
                    "min_child_samples": 33,
                    "min_split_gain": 0.8861115170121059,
                    "colsample_bytree": 0.8446689945438774,
                    "subsample": 0.7140007404693095,
                    "reg_alpha": 1.0256174881780948,
                    "reg_lambda": 7.1849641348728985,
                    "random_state": 724
                }
            },
            2021: {
                724: {
                    "n_estimators": 316,
                    "max_depth": 20,
                    "num_leaves": 122,
                    "learning_rate": 0.05864784590521265,
                    "min_child_samples": 66,
                    "min_split_gain": 0.008812795000978146,
                    "colsample_bytree": 0.1276667131449424,
                    "subsample": 0.6841691149773057,
                    "reg_alpha": 7.74229424829973,
                    "reg_lambda": 4.429297778113121,
                    "random_state": 724
                }
            },
            2022: {
                724: {
                    "n_estimators": 461,
                    "max_depth": 12,
                    "num_leaves": 127,
                    "learning_rate": 0.040843362252636746,
                    "min_child_samples": 49,
                    "min_split_gain": 0.8982233777488002,
                    "colsample_bytree": 0.5481974882915626,
                    "subsample": 0.8755157253994822,
                    "reg_alpha": 1.5083097699772026,
                    "reg_lambda": 9.965208176076544,
                    "random_state": 724
                }
            }
        },
        "3순": {
            2018: {
                724: {
                    "n_estimators": 734,
                    "max_depth": 3,
                    "num_leaves": 20,
                    "learning_rate": 0.1859241585152518,
                    "min_child_samples": 5,
                    "min_split_gain": 0.7443782275124509,
                    "colsample_bytree": 0.5703337575566969,
                    "subsample": 0.616025903449778,
                    "reg_alpha": 6.757418739076651,
                    "reg_lambda": 8.535497908516628,
                    "random_state": 724
                }
            },
            2019: {
                724: {
                    "n_estimators": 727,
                    "max_depth": 10,
                    "num_leaves": 140,
                    "learning_rate": 0.26078562097298286,
                    "min_child_samples": 39,
                    "min_split_gain": 0.6693651403013251,
                    "colsample_bytree": 0.6570513122113372,
                    "subsample": 0.9824809368051572,
                    "reg_alpha": 4.1815923125735495,
                    "reg_lambda": 0.13575595497081538,
                    "random_state": 724
                }
            },
            2020: {
                724: {
                    "n_estimators": 302,
                    "max_depth": 5,
                    "num_leaves": 77,
                    "learning_rate": 0.038197077114263844,
                    "min_child_samples": 43,
                    "min_split_gain": 0.5920249808779021,
                    "colsample_bytree": 0.5948267655507871,
                    "subsample": 0.9533594886803742,
                    "reg_alpha": 5.883566054342359,
                    "reg_lambda": 4.116406897589409,
                    "random_state": 724
                }
            },
            2021: {
                724: {
                    "n_estimators": 442,
                    "max_depth": 4,
                    "num_leaves": 80,
                    "learning_rate": 0.01979903226941752,
                    "min_child_samples": 50,
                    "min_split_gain": 0.9417139833647833,
                    "colsample_bytree": 0.1357660027113984,
                    "subsample": 0.6254866163779498,
                    "reg_alpha": 7.879954157842284,
                    "reg_lambda": 9.237527471033856,
                    "random_state": 724
                }
            },
            2022: {
                724: {
                    "n_estimators": 1712,
                    "max_depth": 7,
                    "num_leaves": 113,
                    "learning_rate": 0.06720899882244163,
                    "min_child_samples": 23,
                    "min_split_gain": 0.6163752266361846,
                    "colsample_bytree": 0.9872258197738184,
                    "subsample": 0.9976216038623017,
                    "reg_alpha": 4.903588086567847,
                    "reg_lambda": 7.554403728641921,
                    "random_state": 724
                }
            }
        }
    },
    "배추": {
        "1순": {
            2018: {
                724: {
                    "n_estimators": 1288,
                    "max_depth": 5,
                    "num_leaves": 61,
                    "learning_rate": 0.05437128911722977,
                    "min_child_samples": 14,
                    "min_split_gain": 0.2507177956928972,
                    "colsample_bytree": 0.3614145327871999,
                    "subsample": 0.9472158302654013,
                    "reg_alpha": 4.7416904937082025,
                    "reg_lambda": 1.4631900049380948,
                    "random_state": 724
                }
            },
            2019: {
                724: {
                    "n_estimators": 2982,
                    "max_depth": 8,
                    "num_leaves": 50,
                    "learning_rate": 0.13303875025489492,
                    "min_child_samples": 10,
                    "min_split_gain": 0.9268571471360578,
                    "colsample_bytree": 0.7771845094192775,
                    "subsample": 0.6439951297758456,
                    "reg_alpha": 6.06494724669048,
                    "reg_lambda": 7.2533830013896505,
                    "random_state": 724
                }
            },
            2020: {
                724: {
                    "n_estimators": 2759,
                    "max_depth": 7,
                    "num_leaves": 112,
                    "learning_rate": 0.08014543458556189,
                    "min_child_samples": 20,
                    "min_split_gain": 0.4375417305322714,
                    "colsample_bytree": 0.3729689533089021,
                    "subsample": 0.8241316984553564,
                    "reg_alpha": 3.5492959260792496,
                    "reg_lambda": 9.436569688454176,
                    "random_state": 724
                }
            },
            2021: {
                724: {
                    "n_estimators": 2344,
                    "max_depth": 20,
                    "num_leaves": 119,
                    "learning_rate": 0.18880210398279645,
                    "min_child_samples": 15,
                    "min_split_gain": 0.15025396102988886,
                    "colsample_bytree": 0.22132022780313287,
                    "subsample": 0.9969724768398283,
                    "reg_alpha": 6.1193305832871285,
                    "reg_lambda": 4.961529658085742,
                    "random_state": 724
                }
            },
            2022: {
                724: {
                    "n_estimators": 1615,
                    "max_depth": 20,
                    "num_leaves": 137,
                    "learning_rate": 0.051067397058457725,
                    "min_child_samples": 8,
                    "min_split_gain": 0.06744231349835554,
                    "colsample_bytree": 0.1370441109254982,
                    "subsample": 0.6469506114815196,
                    "reg_alpha": 4.068541908868682,
                    "reg_lambda": 6.443394049131884,
                    "random_state": 724
                }
            }
        },
        "2순": {
            2018: {
                724: {
                    "n_estimators": 1625,
                    "max_depth": 9,
                    "num_leaves": 138,
                    "learning_rate": 0.18861857100271073,
                    "min_child_samples": 13,
                    "min_split_gain": 0.45064342268690744,
                    "colsample_bytree": 0.35876783438398474,
                    "subsample": 0.8502012770778826,
                    "reg_alpha": 6.730838415047526,
                    "reg_lambda": 9.262301976036515,
                    "random_state": 724
                }
            },
            2019: {
                724: {
                    "n_estimators": 1446,
                    "max_depth": 17,
                    "num_leaves": 63,
                    "learning_rate": 0.047954543396829204,
                    "min_child_samples": 12,
                    "min_split_gain": 0.894107990713245,
                    "colsample_bytree": 0.5475175247727464,
                    "subsample": 0.8238934788295674,
                    "reg_alpha": 3.732695110736458,
                    "reg_lambda": 4.433286327814227,
                    "random_state": 724
                }
            },
            2020: {
                724: {
                    "n_estimators": 2186,
                    "max_depth": 12,
                    "num_leaves": 112,
                    "learning_rate": 0.17060974452583522,
                    "min_child_samples": 5,
                    "min_split_gain": 0.18855729112314804,
                    "colsample_bytree": 0.5681274807340617,
                    "subsample": 0.6721811262567319,
                    "reg_alpha": 8.863528867400849,
                    "reg_lambda": 1.2958623725462581,
                    "random_state": 724
                }
            },
            2021: {
                724: {
                    "n_estimators": 743,
                    "max_depth": 12,
                    "num_leaves": 140,
                    "learning_rate": 0.03957534478591516,
                    "min_child_samples": 13,
                    "min_split_gain": 0.26754510899932615,
                    "colsample_bytree": 0.10456231963788876,
                    "subsample": 0.8321289568367797,
                    "reg_alpha": 6.0794219677763195,
                    "reg_lambda": 6.568053631823488,
                    "random_state": 724
                }
            },
            2022: {
                724: {
                    "n_estimators": 1718,
                    "max_depth": 20,
                    "num_leaves": 145,
                    "learning_rate": 0.2610129920560204,
                    "min_child_samples": 24,
                    "min_split_gain": 0.5727676945171719,
                    "colsample_bytree": 0.155838425790967,
                    "subsample": 0.7516279254791979,
                    "reg_alpha": 4.610341113293735,
                    "reg_lambda": 2.662249921647893,
                    "random_state": 724
                }
            }
        },
        "3순": {
            2018: {
                724: {
                    "n_estimators": 1115,
                    "max_depth": 9,
                    "num_leaves": 129,
                    "learning_rate": 0.08190614871006169,
                    "min_child_samples": 11,
                    "min_split_gain": 0.10894404572886997,
                    "colsample_bytree": 0.11509821586747782,
                    "subsample": 0.7920814303926111,
                    "reg_alpha": 9.132881912517446,
                    "reg_lambda": 8.082195777578795,
                    "random_state": 724
                }
            },
            2019: {
                724: {
                    "n_estimators": 549,
                    "max_depth": 6,
                    "num_leaves": 150,
                    "learning_rate": 0.15573321139155294,
                    "min_child_samples": 12,
                    "min_split_gain": 0.4131991266619709,
                    "colsample_bytree": 0.6998279183604453,
                    "subsample": 0.8569783422488777,
                    "reg_alpha": 0.5722236129016336,
                    "reg_lambda": 6.413423783850889,
                    "random_state": 724
                }
            },
            2020: {
                724: {
                    "n_estimators": 447,
                    "max_depth": 19,
                    "num_leaves": 119,
                    "learning_rate": 0.20528863393909594,
                    "min_child_samples": 5,
                    "min_split_gain": 0.002943277688478768,
                    "colsample_bytree": 0.3655535960421358,
                    "subsample": 0.8951284596617232,
                    "reg_alpha": 7.491360454978178,
                    "reg_lambda": 9.230652789477347,
                    "random_state": 724
                }
            },
            2021: {
                724: {
                    "n_estimators": 1898,
                    "max_depth": 18,
                    "num_leaves": 89,
                    "learning_rate": 0.05716922262111397,
                    "min_child_samples": 14,
                    "min_split_gain": 0.0024499159533911685,
                    "colsample_bytree": 0.11117117594335192,
                    "subsample": 0.8064943608830302,
                    "reg_alpha": 3.3536527217301795,
                    "reg_lambda": 3.2664314527350777,
                    "random_state": 724
                }
            },
            2022: {
                724: {
                    "n_estimators": 3000,
                    "max_depth": 11,
                    "num_leaves": 134,
                    "learning_rate": 0.18693415601372607,
                    "min_child_samples": 23,
                    "min_split_gain": 0.45283430952687365,
                    "colsample_bytree": 0.26004371973303564,
                    "subsample": 0.756988620015125,
                    "reg_alpha": 3.6425362081466552,
                    "reg_lambda": 5.8674690751836085,
                    "random_state": 724
                }
            }
        }
    },
    "대파(일반)": {
        "1순": {
            2018: {
                724: {
                    "n_estimators": 915,
                    "max_depth": 3,
                    "num_leaves": 75,
                    "learning_rate": 0.2755200095199343,
                    "min_child_samples": 11,
                    "min_split_gain": 0.1723539916593918,
                    "colsample_bytree": 0.26630784702527105,
                    "subsample": 0.6828853579874045,
                    "reg_alpha": 7.406023425915772,
                    "reg_lambda": 8.713391010966163,
                    "random_state": 724
                }
            },
            2019: {
                724: {
                    "n_estimators": 1099,
                    "max_depth": 9,
                    "num_leaves": 78,
                    "learning_rate": 0.01138087723784853,
                    "min_child_samples": 22,
                    "min_split_gain": 0.13658746897299567,
                    "colsample_bytree": 0.29174269916609735,
                    "subsample": 0.7263960428545815,
                    "reg_alpha": 8.300190921548793,
                    "reg_lambda": 9.386418156665117,
                    "random_state": 724
                }
            },
            2020: {
                724: {
                    "n_estimators": 2152,
                    "max_depth": 5,
                    "num_leaves": 90,
                    "learning_rate": 0.25704314957314256,
                    "min_child_samples": 5,
                    "min_split_gain": 0.18892041646001945,
                    "colsample_bytree": 0.5818381619450579,
                    "subsample": 0.6671405580172897,
                    "reg_alpha": 4.319681387738347,
                    "reg_lambda": 0.6865030316853549,
                    "random_state": 724
                }
            },
            2021: {
                724: {
                    "n_estimators": 531,
                    "max_depth": 17,
                    "num_leaves": 44,
                    "learning_rate": 0.14854210285831176,
                    "min_child_samples": 11,
                    "min_split_gain": 0.2267047314650632,
                    "colsample_bytree": 0.7137577142542499,
                    "subsample": 0.6502742700584809,
                    "reg_alpha": 7.87747356832702,
                    "reg_lambda": 5.757637779282252,
                    "random_state": 724
                }
            },
            2022: {
                724: {
                    "n_estimators": 1094,
                    "max_depth": 13,
                    "num_leaves": 102,
                    "learning_rate": 0.01267955747873004,
                    "min_child_samples": 15,
                    "min_split_gain": 0.8292129186019435,
                    "colsample_bytree": 0.3622994200289589,
                    "subsample": 0.5626531829591873,
                    "reg_alpha": 4.541870192034362,
                    "reg_lambda": 5.223483063605906,
                    "random_state": 724
                }
            }
        },
        "2순": {
            2018: {
                724: {
                    "n_estimators": 478,
                    "max_depth": 8,
                    "num_leaves": 68,
                    "learning_rate": 0.15255673882036655,
                    "min_child_samples": 13,
                    "min_split_gain": 0.3775832273328433,
                    "colsample_bytree": 0.9233228835483223,
                    "subsample": 0.5692058366901391,
                    "reg_alpha": 4.0683904372376976,
                    "reg_lambda": 5.242238795268011,
                    "random_state": 724
                }
            },
            2019: {
                724: {
                    "n_estimators": 2423,
                    "max_depth": 17,
                    "num_leaves": 31,
                    "learning_rate": 0.01678609858940428,
                    "min_child_samples": 14,
                    "min_split_gain": 0.5827439581199595,
                    "colsample_bytree": 0.4922061796345526,
                    "subsample": 0.6957855512901185,
                    "reg_alpha": 6.334133405135081,
                    "reg_lambda": 1.1364622120169732,
                    "random_state": 724
                }
            },
            2020: {
                724: {
                    "n_estimators": 1540,
                    "max_depth": 15,
                    "num_leaves": 60,
                    "learning_rate": 0.09891703731031973,
                    "min_child_samples": 54,
                    "min_split_gain": 0.21872909461949894,
                    "colsample_bytree": 0.9125186043118358,
                    "subsample": 0.6870803150842781,
                    "reg_alpha": 5.036693720090878,
                    "reg_lambda": 5.50307646443718,
                    "random_state": 724
                }
            },
            2021: {
                724: {
                    "n_estimators": 1659,
                    "max_depth": 10,
                    "num_leaves": 56,
                    "learning_rate": 0.14196894613750202,
                    "min_child_samples": 21,
                    "min_split_gain": 0.33079513118100656,
                    "colsample_bytree": 0.3735747021269079,
                    "subsample": 0.5206520491745776,
                    "reg_alpha": 1.1241795213093904,
                    "reg_lambda": 9.338987867297066,
                    "random_state": 724
                }
            },
            2022: {
                724: {
                    "n_estimators": 1778,
                    "max_depth": 12,
                    "num_leaves": 40,
                    "learning_rate": 0.2404386266327172,
                    "min_child_samples": 17,
                    "min_split_gain": 0.7233741609913146,
                    "colsample_bytree": 0.3591916932184335,
                    "subsample": 0.689401680187583,
                    "reg_alpha": 2.795934630678832,
                    "reg_lambda": 5.357751831699911,
                    "random_state": 724
                }
            }
        },
        "3순": {
            2018: {
                724: {
                    "n_estimators": 664,
                    "max_depth": 16,
                    "num_leaves": 90,
                    "learning_rate": 0.03536187862801591,
                    "min_child_samples": 17,
                    "min_split_gain": 0.1600822867032634,
                    "colsample_bytree": 0.31769214167013254,
                    "subsample": 0.7033235827528717,
                    "reg_alpha": 2.440373481451651,
                    "reg_lambda": 0.6444627759052863,
                    "random_state": 724
                }
            },
            2019: {
                724: {
                    "n_estimators": 2329,
                    "max_depth": 5,
                    "num_leaves": 28,
                    "learning_rate": 0.17765415748487928,
                    "min_child_samples": 13,
                    "min_split_gain": 0.794823074988899,
                    "colsample_bytree": 0.6440351262576225,
                    "subsample": 0.8507051826553704,
                    "reg_alpha": 3.7103981612917822,
                    "reg_lambda": 5.065162436551272,
                    "random_state": 724
                }
            },
            2020: {
                724: {
                    "n_estimators": 339,
                    "max_depth": 19,
                    "num_leaves": 109,
                    "learning_rate": 0.12830302149794456,
                    "min_child_samples": 33,
                    "min_split_gain": 0.8924459728499138,
                    "colsample_bytree": 0.14213163502928916,
                    "subsample": 0.9485136933634498,
                    "reg_alpha": 1.2129792562115795,
                    "reg_lambda": 8.414802105261233,
                    "random_state": 724
                }
            },
            2021: {
                724: {
                    "n_estimators": 2520,
                    "max_depth": 15,
                    "num_leaves": 123,
                    "learning_rate": 0.012759560301591005,
                    "min_child_samples": 9,
                    "min_split_gain": 0.7236882290119069,
                    "colsample_bytree": 0.2710261767540143,
                    "subsample": 0.6010690529667133,
                    "reg_alpha": 1.750410262747416,
                    "reg_lambda": 8.093589369535952,
                    "random_state": 724
                }
            },
            2022: {
                724: {
                    "n_estimators": 886,
                    "max_depth": 14,
                    "num_leaves": 72,
                    "learning_rate": 0.14191054810822223,
                    "min_child_samples": 19,
                    "min_split_gain": 0.7546579750954211,
                    "colsample_bytree": 0.8270811265940041,
                    "subsample": 0.9284543557421284,
                    "reg_alpha": 6.560968597249294,
                    "reg_lambda": 4.7812692283480756,
                    "random_state": 724
                }
            }
        }
    },
    "건고추": {
        "1순": {
            2018: {
                724: {
                    "n_estimators": 1978,
                    "max_depth": 17,
                    "num_leaves": 136,
                    "learning_rate": 0.25530807815655227,
                    "min_child_samples": 30,
                    "min_split_gain": 0.004168890759429722,
                    "colsample_bytree": 0.4873307122949527,
                    "subsample": 0.9181217610908115,
                    "reg_alpha": 0.07475595842164928,
                    "reg_lambda": 3.297113621030069,
                    "random_state": 724
                }
            },
            2019: {
                724: {
                    "n_estimators": 2865,
                    "max_depth": 10,
                    "num_leaves": 149,
                    "learning_rate": 0.0890742387804514,
                    "min_child_samples": 19,
                    "min_split_gain": 0.7868649005325302,
                    "colsample_bytree": 0.34501162058372403,
                    "subsample": 0.5721632689059043,
                    "reg_alpha": 4.68394647682174,
                    "reg_lambda": 2.172534929161671,
                    "random_state": 724
                }
            },
            2020: {
                724: {
                    "n_estimators": 1081,
                    "max_depth": 12,
                    "num_leaves": 43,
                    "learning_rate": 0.24239401988358297,
                    "min_child_samples": 10,
                    "min_split_gain": 0.15458285069824457,
                    "colsample_bytree": 0.3887270809083718,
                    "subsample": 0.8719821089741816,
                    "reg_alpha": 8.756251445895586,
                    "reg_lambda": 0.01896671245922632,
                    "random_state": 724
                }
            },
            2021: {
                724: {
                    "n_estimators": 2454,
                    "max_depth": 13,
                    "num_leaves": 78,
                    "learning_rate": 0.09467618090992147,
                    "min_child_samples": 11,
                    "min_split_gain": 0.5382162062444774,
                    "colsample_bytree": 0.6017960664120294,
                    "subsample": 0.7315607360448791,
                    "reg_alpha": 5.474418224768519,
                    "reg_lambda": 7.7701924381140515,
                    "random_state": 724
                }
            },
            2022: {
                724: {
                    "n_estimators": 1810,
                    "max_depth": 10,
                    "num_leaves": 77,
                    "learning_rate": 0.027385987263031354,
                    "min_child_samples": 23,
                    "min_split_gain": 0.4004999477261171,
                    "colsample_bytree": 0.9971173922699614,
                    "subsample": 0.8337136543211433,
                    "reg_alpha": 7.111441633679647,
                    "reg_lambda": 6.403862359637419,
                    "random_state": 724
                }
            }
        },
        "2순": {
            2018: {
                724: {
                    "n_estimators": 1680,
                    "max_depth": 17,
                    "num_leaves": 28,
                    "learning_rate": 0.17945356511863908,
                    "min_child_samples": 26,
                    "min_split_gain": 0.861878178953978,
                    "colsample_bytree": 0.9163163624653263,
                    "subsample": 0.576549773965579,
                    "reg_alpha": 6.197714836148259,
                    "reg_lambda": 2.1068274802555926,
                    "random_state": 724
                }
            },
            2019: {
                724: {
                    "n_estimators": 521,
                    "max_depth": 16,
                    "num_leaves": 110,
                    "learning_rate": 0.02741996573865757,
                    "min_child_samples": 40,
                    "min_split_gain": 0.40778709372465705,
                    "colsample_bytree": 0.29466826136254726,
                    "subsample": 0.5557501256350099,
                    "reg_alpha": 9.041525544522552,
                    "reg_lambda": 7.670554068936848,
                    "random_state": 724
                }
            },
            2020: {
                724: {
                    "n_estimators": 997,
                    "max_depth": 14,
                    "num_leaves": 27,
                    "learning_rate": 0.06377734258591775,
                    "min_child_samples": 28,
                    "min_split_gain": 0.2923832630383172,
                    "colsample_bytree": 0.6971648255601116,
                    "subsample": 0.9365099974888884,
                    "reg_alpha": 2.137668924971916,
                    "reg_lambda": 3.9660806125841095,
                    "random_state": 724
                }
            },
            2021: {
                724: {
                    "n_estimators": 448,
                    "max_depth": 8,
                    "num_leaves": 139,
                    "learning_rate": 0.29659125534139813,
                    "min_child_samples": 6,
                    "min_split_gain": 0.4975823011542556,
                    "colsample_bytree": 0.45411148687181235,
                    "subsample": 0.6934953713146255,
                    "reg_alpha": 2.0695093965753077,
                    "reg_lambda": 0.021750792292482002,
                    "random_state": 724
                }
            },
            2022: {
                724: {
                    "n_estimators": 1019,
                    "max_depth": 5,
                    "num_leaves": 134,
                    "learning_rate": 0.11786861623287842,
                    "min_child_samples": 31,
                    "min_split_gain": 0.9838496690559779,
                    "colsample_bytree": 0.27788528735864426,
                    "subsample": 0.5892834892057761,
                    "reg_alpha": 1.2674169034398295,
                    "reg_lambda": 2.5119052250807354,
                    "random_state": 724
                }
            }
        },
        "3순": {
            2018: {
                724: {
                    "n_estimators": 345,
                    "max_depth": 16,
                    "num_leaves": 150,
                    "learning_rate": 0.012170146501920866,
                    "min_child_samples": 70,
                    "min_split_gain": 0.4770292263905513,
                    "colsample_bytree": 0.35599655634173916,
                    "subsample": 0.8212320022129505,
                    "reg_alpha": 0.10440974633409161,
                    "reg_lambda": 6.704916945390442,
                    "random_state": 724
                }
            },
            2019: {
                724: {
                    "n_estimators": 1142,
                    "max_depth": 4,
                    "num_leaves": 36,
                    "learning_rate": 0.08366534199186475,
                    "min_child_samples": 9,
                    "min_split_gain": 0.5185882945472509,
                    "colsample_bytree": 0.3247141504690748,
                    "subsample": 0.5242974324034992,
                    "reg_alpha": 1.840781932125421,
                    "reg_lambda": 0.30830022338616425,
                    "random_state": 724
                }
            },
            2020: {
                724: {
                    "n_estimators": 1229,
                    "max_depth": 20,
                    "num_leaves": 43,
                    "learning_rate": 0.19082270123730105,
                    "min_child_samples": 25,
                    "min_split_gain": 0.7936814650372113,
                    "colsample_bytree": 0.9323623408135003,
                    "subsample": 0.8369767631499208,
                    "reg_alpha": 6.213205039467133,
                    "reg_lambda": 4.487073170253012,
                    "random_state": 724
                }
            },
            2021: {
                724: {
                    "n_estimators": 932,
                    "max_depth": 12,
                    "num_leaves": 83,
                    "learning_rate": 0.1801209601439777,
                    "min_child_samples": 62,
                    "min_split_gain": 0.2162027981394104,
                    "colsample_bytree": 0.7241072913614327,
                    "subsample": 0.9474945663426708,
                    "reg_alpha": 3.068366983119068,
                    "reg_lambda": 8.976740396816718,
                    "random_state": 724
                }
            },
            2022: {
                724: {
                    "n_estimators": 1559,
                    "max_depth": 8,
                    "num_leaves": 100,
                    "learning_rate": 0.14070037106191483,
                    "min_child_samples": 25,
                    "min_split_gain": 0.37089754592078705,
                    "colsample_bytree": 0.7290359821688296,
                    "subsample": 0.6791893707434494,
                    "reg_alpha": 4.210806589437073,
                    "reg_lambda": 2.8827074963757564,
                    "random_state": 724
                }
            }
        }
    },
    "깐마늘(국산)": {
        "1순": {
            2018: {
                724: {
                    "n_estimators": 823,
                    "max_depth": 20,
                    "num_leaves": 51,
                    "learning_rate": 0.2828868064836877,
                    "min_child_samples": 29,
                    "min_split_gain": 0.29659271878491317,
                    "colsample_bytree": 0.6148227891631906,
                    "subsample": 0.7673815389937133,
                    "reg_alpha": 1.1867251253571607,
                    "reg_lambda": 3.5708607443915255,
                    "random_state": 724
                }
            },
            2019: {
                724: {
                    "n_estimators": 834,
                    "max_depth": 16,
                    "num_leaves": 54,
                    "learning_rate": 0.15492862846464864,
                    "min_child_samples": 88,
                    "min_split_gain": 0.5030508860103747,
                    "colsample_bytree": 0.6758739328171006,
                    "subsample": 0.9606936533481929,
                    "reg_alpha": 0.1918197362352203,
                    "reg_lambda": 2.4350585143600414,
                    "random_state": 724
                }
            },
            2020: {
                724: {
                    "n_estimators": 1888,
                    "max_depth": 14,
                    "num_leaves": 129,
                    "learning_rate": 0.05502127846790547,
                    "min_child_samples": 97,
                    "min_split_gain": 0.3738793771789831,
                    "colsample_bytree": 0.8579216463114947,
                    "subsample": 0.8976829535612337,
                    "reg_alpha": 6.957369138734365,
                    "reg_lambda": 8.086327047470562,
                    "random_state": 724
                }
            },
            2021: {
                724: {
                    "n_estimators": 2142,
                    "max_depth": 6,
                    "num_leaves": 60,
                    "learning_rate": 0.24301943469112414,
                    "min_child_samples": 30,
                    "min_split_gain": 0.6403578183667666,
                    "colsample_bytree": 0.5122615613968032,
                    "subsample": 0.6169727834883002,
                    "reg_alpha": 8.192301915443142,
                    "reg_lambda": 2.3306702037308016,
                    "random_state": 724
                }
            },
            2022: {
                724: {
                    "n_estimators": 2046,
                    "max_depth": 10,
                    "num_leaves": 47,
                    "learning_rate": 0.23955719957636687,
                    "min_child_samples": 23,
                    "min_split_gain": 0.9207246042570882,
                    "colsample_bytree": 0.661839421530458,
                    "subsample": 0.8418627476813944,
                    "reg_alpha": 9.957046341824753,
                    "reg_lambda": 3.260566254929696,
                    "random_state": 724
                }
            }
        },
        "2순": {
            2018: {
                724: {
                    "n_estimators": 393,
                    "max_depth": 17,
                    "num_leaves": 148,
                    "learning_rate": 0.22645409854356424,
                    "min_child_samples": 86,
                    "min_split_gain": 0.7060453531791007,
                    "colsample_bytree": 0.8283329627824471,
                    "subsample": 0.6662716974852616,
                    "reg_alpha": 5.052098876878336,
                    "reg_lambda": 5.178444736926901,
                    "random_state": 724
                }
            },
            2019: {
                724: {
                    "n_estimators": 2755,
                    "max_depth": 14,
                    "num_leaves": 103,
                    "learning_rate": 0.14141388242234346,
                    "min_child_samples": 73,
                    "min_split_gain": 0.6724007796865955,
                    "colsample_bytree": 0.8598041517843737,
                    "subsample": 0.7063498064385272,
                    "reg_alpha": 6.369872598967619,
                    "reg_lambda": 2.453672668404611,
                    "random_state": 724
                }
            },
            2020: {
                724: {
                    "n_estimators": 652,
                    "max_depth": 15,
                    "num_leaves": 101,
                    "learning_rate": 0.17182562319281228,
                    "min_child_samples": 89,
                    "min_split_gain": 0.5701576801115494,
                    "colsample_bytree": 0.31994237160977224,
                    "subsample": 0.8144780628998671,
                    "reg_alpha": 2.294224582131693,
                    "reg_lambda": 2.1375145660252013,
                    "random_state": 724
                }
            },
            2021: {
                724: {
                    "n_estimators": 973,
                    "max_depth": 3,
                    "num_leaves": 150,
                    "learning_rate": 0.1776465307799442,
                    "min_child_samples": 21,
                    "min_split_gain": 0.7963796354615363,
                    "colsample_bytree": 0.8820670628814724,
                    "subsample": 0.8426668019617671,
                    "reg_alpha": 6.7400324474880975,
                    "reg_lambda": 6.135491978667682,
                    "random_state": 724
                }
            },
            2022: {
                724: {
                    "n_estimators": 1114,
                    "max_depth": 3,
                    "num_leaves": 57,
                    "learning_rate": 0.17978586084239354,
                    "min_child_samples": 27,
                    "min_split_gain": 0.2975864971053306,
                    "colsample_bytree": 0.2332325565573046,
                    "subsample": 0.6973301649493409,
                    "reg_alpha": 6.004528253847507,
                    "reg_lambda": 4.0024602807339456,
                    "random_state": 724
                }
            }
        },
        "3순": {
            2018: {
                724: {
                    "n_estimators": 2008,
                    "max_depth": 14,
                    "num_leaves": 117,
                    "learning_rate": 0.04463461109818993,
                    "min_child_samples": 91,
                    "min_split_gain": 0.04958645194272926,
                    "colsample_bytree": 0.2749685709353836,
                    "subsample": 0.6772888482847796,
                    "reg_alpha": 0.17705401659824926,
                    "reg_lambda": 2.504374349181914,
                    "random_state": 724
                }
            },
            2019: {
                724: {
                    "n_estimators": 1246,
                    "max_depth": 5,
                    "num_leaves": 26,
                    "learning_rate": 0.013631029669608236,
                    "min_child_samples": 90,
                    "min_split_gain": 0.9532298941000092,
                    "colsample_bytree": 0.8828470831846816,
                    "subsample": 0.9824195108177981,
                    "reg_alpha": 5.5630775433580135,
                    "reg_lambda": 6.383305089138197,
                    "random_state": 724
                }
            },
            2020: {
                724: {
                    "n_estimators": 2357,
                    "max_depth": 19,
                    "num_leaves": 53,
                    "learning_rate": 0.07181368889319088,
                    "min_child_samples": 75,
                    "min_split_gain": 0.6992480440617794,
                    "colsample_bytree": 0.9825022271136247,
                    "subsample": 0.8930586940192722,
                    "reg_alpha": 3.271625782016123,
                    "reg_lambda": 7.675248421222011,
                    "random_state": 724
                }
            },
            2021: {
                724: {
                    "n_estimators": 2525,
                    "max_depth": 5,
                    "num_leaves": 36,
                    "learning_rate": 0.24847499123669842,
                    "min_child_samples": 21,
                    "min_split_gain": 0.6996270565066641,
                    "colsample_bytree": 0.22595735884522777,
                    "subsample": 0.7310896034910627,
                    "reg_alpha": 5.1054313553459645,
                    "reg_lambda": 0.271095991119673,
                    "random_state": 724
                }
            },
            2022: {
                724: {
                    "n_estimators": 2497,
                    "max_depth": 18,
                    "num_leaves": 126,
                    "learning_rate": 0.02566411562559051,
                    "min_child_samples": 17,
                    "min_split_gain": 0.544501940531187,
                    "colsample_bytree": 0.3995448578019155,
                    "subsample": 0.764757040750697,
                    "reg_alpha": 7.5292521655666,
                    "reg_lambda": 4.100993804725799,
                    "random_state": 724
                }
            }
        }
    },
    "사과": {
        "1순": {
            2018: {
                724: {
                    "n_estimators": 1514,
                    "max_depth": 17,
                    "num_leaves": 118,
                    "learning_rate": 0.03732236633120435,
                    "min_child_samples": 28,
                    "min_split_gain": 0.5968236383213065,
                    "colsample_bytree": 0.13516969299326906,
                    "subsample": 0.6680479327431345,
                    "reg_alpha": 6.013690461582471,
                    "reg_lambda": 0.5397704177352525,
                    "random_state": 724
                }
            },
            2019: {
                724: {
                    "n_estimators": 2220,
                    "max_depth": 7,
                    "num_leaves": 52,
                    "learning_rate": 0.1362357560619629,
                    "min_child_samples": 34,
                    "min_split_gain": 0.9906134101300392,
                    "colsample_bytree": 0.8139104980214373,
                    "subsample": 0.7923122097993687,
                    "reg_alpha": 1.6526718582075508,
                    "reg_lambda": 6.469935872361959,
                    "random_state": 724
                }
            },
            2020: {
                724: {
                    "n_estimators": 1009,
                    "max_depth": 13,
                    "num_leaves": 22,
                    "learning_rate": 0.17446553678990112,
                    "min_child_samples": 37,
                    "min_split_gain": 0.4973703058704214,
                    "colsample_bytree": 0.41126281589883584,
                    "subsample": 0.9196280256822268,
                    "reg_alpha": 1.759546911404355,
                    "reg_lambda": 6.826393602710317,
                    "random_state": 724
                }
            },
            2021: {
                724: {
                    "n_estimators": 2658,
                    "max_depth": 20,
                    "num_leaves": 53,
                    "learning_rate": 0.028213194753209405,
                    "min_child_samples": 15,
                    "min_split_gain": 0.25377449261497986,
                    "colsample_bytree": 0.6978793564834296,
                    "subsample": 0.8395909925756988,
                    "reg_alpha": 4.8467944015344555,
                    "reg_lambda": 3.9609983609358097,
                    "random_state": 724
                }
            },
            2022: {
                724: {
                    "n_estimators": 1035,
                    "max_depth": 17,
                    "num_leaves": 114,
                    "learning_rate": 0.24061678805024497,
                    "min_child_samples": 65,
                    "min_split_gain": 0.5502902130457202,
                    "colsample_bytree": 0.4755437508600101,
                    "subsample": 0.6555341815073381,
                    "reg_alpha": 2.276976686940994,
                    "reg_lambda": 8.212728899013829,
                    "random_state": 724
                }
            }
        },
        "2순": {
            2018: {
                724: {
                    "n_estimators": 1417,
                    "max_depth": 7,
                    "num_leaves": 47,
                    "learning_rate": 0.05071549354139885,
                    "min_child_samples": 36,
                    "min_split_gain": 0.9896240112798782,
                    "colsample_bytree": 0.866102406861893,
                    "subsample": 0.842044186918111,
                    "reg_alpha": 4.0217651533066645,
                    "reg_lambda": 5.811264102460939,
                    "random_state": 724
                }
            },
            2019: {
                724: {
                    "n_estimators": 1695,
                    "max_depth": 18,
                    "num_leaves": 62,
                    "learning_rate": 0.16842480037284244,
                    "min_child_samples": 39,
                    "min_split_gain": 0.26178721984363007,
                    "colsample_bytree": 0.868841896930318,
                    "subsample": 0.821205375542599,
                    "reg_alpha": 5.5237433492445325,
                    "reg_lambda": 9.22356817896084,
                    "random_state": 724
                }
            },
            2020: {
                724: {
                    "n_estimators": 2129,
                    "max_depth": 15,
                    "num_leaves": 109,
                    "learning_rate": 0.2661918434489972,
                    "min_child_samples": 29,
                    "min_split_gain": 0.3214424656841106,
                    "colsample_bytree": 0.465444649621536,
                    "subsample": 0.630954717515709,
                    "reg_alpha": 7.044916860660065,
                    "reg_lambda": 2.6832795792480795,
                    "random_state": 724
                }
            },
            2021: {
                724: {
                    "n_estimators": 1203,
                    "max_depth": 7,
                    "num_leaves": 76,
                    "learning_rate": 0.23867482840059143,
                    "min_child_samples": 19,
                    "min_split_gain": 0.8332355113691238,
                    "colsample_bytree": 0.2614214644184343,
                    "subsample": 0.8664145839134743,
                    "reg_alpha": 5.7176921018008215,
                    "reg_lambda": 2.8352171276205507,
                    "random_state": 724
                }
            },
            2022: {
                724: {
                    "n_estimators": 1037,
                    "max_depth": 14,
                    "num_leaves": 144,
                    "learning_rate": 0.023965543634606744,
                    "min_child_samples": 62,
                    "min_split_gain": 0.538368781220851,
                    "colsample_bytree": 0.8214741719328901,
                    "subsample": 0.8315534347868365,
                    "reg_alpha": 8.64303750281902,
                    "reg_lambda": 4.1414709704345425,
                    "random_state": 724
                }
            }
        },
        "3순": {
            2018: {
                724: {
                    "n_estimators": 1242,
                    "max_depth": 17,
                    "num_leaves": 55,
                    "learning_rate": 0.2931548926304034,
                    "min_child_samples": 31,
                    "min_split_gain": 0.0027429344668626343,
                    "colsample_bytree": 0.38205424841248037,
                    "subsample": 0.7952989225503067,
                    "reg_alpha": 3.8198229067400176,
                    "reg_lambda": 9.850360935800914,
                    "random_state": 724
                }
            },
            2019: {
                724: {
                    "n_estimators": 1247,
                    "max_depth": 17,
                    "num_leaves": 38,
                    "learning_rate": 0.026198091707657394,
                    "min_child_samples": 42,
                    "min_split_gain": 0.6305592377663756,
                    "colsample_bytree": 0.8948834099915479,
                    "subsample": 0.5058998633363688,
                    "reg_alpha": 9.32572504869449,
                    "reg_lambda": 6.021729489549505,
                    "random_state": 724
                }
            },
            2020: {
                724: {
                    "n_estimators": 2261,
                    "max_depth": 17,
                    "num_leaves": 120,
                    "learning_rate": 0.016587362542156453,
                    "min_child_samples": 32,
                    "min_split_gain": 0.6299127075129907,
                    "colsample_bytree": 0.7752220610236213,
                    "subsample": 0.694736521123942,
                    "reg_alpha": 9.009300179671389,
                    "reg_lambda": 6.3536349336234785,
                    "random_state": 724
                }
            },
            2021: {
                724: {
                    "n_estimators": 1906,
                    "max_depth": 17,
                    "num_leaves": 40,
                    "learning_rate": 0.05889446090826285,
                    "min_child_samples": 41,
                    "min_split_gain": 0.6575080939025538,
                    "colsample_bytree": 0.25510100411413567,
                    "subsample": 0.9090919422265734,
                    "reg_alpha": 0.17998289034049708,
                    "reg_lambda": 7.442728554380408,
                    "random_state": 724
                }
            },
            2022: {
                724: {
                    "n_estimators": 1956,
                    "max_depth": 15,
                    "num_leaves": 28,
                    "learning_rate": 0.29876499062793205,
                    "min_child_samples": 13,
                    "min_split_gain": 0.5316756328791072,
                    "colsample_bytree": 0.3448731990786619,
                    "subsample": 0.6134631362637967,
                    "reg_alpha": 7.9854733962986675,
                    "reg_lambda": 7.36025919797873,
                    "random_state": 724
                }
            }
        }
    },
    "배": {
        "1순": {
            2018: {
                724: {
                    "n_estimators": 1903,
                    "max_depth": 15,
                    "num_leaves": 76,
                    "learning_rate": 0.16454247834944935,
                    "min_child_samples": 9,
                    "min_split_gain": 0.7468832222631732,
                    "colsample_bytree": 0.9986696354052023,
                    "subsample": 0.7786881948030467,
                    "reg_alpha": 6.131794973147269,
                    "reg_lambda": 4.354350067513306,
                    "random_state": 724
                }
            },
            2019: {
                724: {
                    "n_estimators": 2104,
                    "max_depth": 11,
                    "num_leaves": 39,
                    "learning_rate": 0.0925929638100275,
                    "min_child_samples": 9,
                    "min_split_gain": 0.7306035171883386,
                    "colsample_bytree": 0.7520434851414856,
                    "subsample": 0.8906585167208185,
                    "reg_alpha": 1.3838920629610225,
                    "reg_lambda": 6.305162856029374,
                    "random_state": 724
                }
            },
            2020: {
                724: {
                    "n_estimators": 2407,
                    "max_depth": 12,
                    "num_leaves": 138,
                    "learning_rate": 0.21241744097817272,
                    "min_child_samples": 13,
                    "min_split_gain": 0.6304134579548717,
                    "colsample_bytree": 0.8939834618351368,
                    "subsample": 0.6828468673878397,
                    "reg_alpha": 0.19075056886226482,
                    "reg_lambda": 1.219071535462882,
                    "random_state": 724
                }
            },
            2021: {
                724: {
                    "n_estimators": 2240,
                    "max_depth": 7,
                    "num_leaves": 105,
                    "learning_rate": 0.12238654009327266,
                    "min_child_samples": 32,
                    "min_split_gain": 0.8705269797016798,
                    "colsample_bytree": 0.9146011191574057,
                    "subsample": 0.8992715921903636,
                    "reg_alpha": 3.315480501183301,
                    "reg_lambda": 8.323276380908073,
                    "random_state": 724
                }
            },
            2022: {
                724: {
                    "n_estimators": 1989,
                    "max_depth": 20,
                    "num_leaves": 95,
                    "learning_rate": 0.20203482346350282,
                    "min_child_samples": 58,
                    "min_split_gain": 0.7865339222249381,
                    "colsample_bytree": 0.5838110174147106,
                    "subsample": 0.9334297225184123,
                    "reg_alpha": 2.3050934373377814,
                    "reg_lambda": 3.6195022072767133,
                    "random_state": 724
                }
            }
        },
        "2순": {
            2018: {
                724: {
                    "n_estimators": 1813,
                    "max_depth": 18,
                    "num_leaves": 26,
                    "learning_rate": 0.14292978253755886,
                    "min_child_samples": 68,
                    "min_split_gain": 0.18452591060480816,
                    "colsample_bytree": 0.42066484277658284,
                    "subsample": 0.5332461902475704,
                    "reg_alpha": 1.1156487233647128,
                    "reg_lambda": 4.281606829508293,
                    "random_state": 724
                }
            },
            2019: {
                724: {
                    "n_estimators": 1579,
                    "max_depth": 16,
                    "num_leaves": 31,
                    "learning_rate": 0.241517996113279,
                    "min_child_samples": 8,
                    "min_split_gain": 0.3055993247038973,
                    "colsample_bytree": 0.9386958541623541,
                    "subsample": 0.7992471357588335,
                    "reg_alpha": 9.395358895241866,
                    "reg_lambda": 7.230071633361343,
                    "random_state": 724
                }
            },
            2020: {
                724: {
                    "n_estimators": 2917,
                    "max_depth": 9,
                    "num_leaves": 128,
                    "learning_rate": 0.26695812769414456,
                    "min_child_samples": 28,
                    "min_split_gain": 0.06115866304619533,
                    "colsample_bytree": 0.10171372839837015,
                    "subsample": 0.8359505851206699,
                    "reg_alpha": 7.540918519553873,
                    "reg_lambda": 2.165956853502993,
                    "random_state": 724
                }
            },
            2021: {
                724: {
                    "n_estimators": 1768,
                    "max_depth": 13,
                    "num_leaves": 77,
                    "learning_rate": 0.04459949019279322,
                    "min_child_samples": 31,
                    "min_split_gain": 0.9035098484686384,
                    "colsample_bytree": 0.5020825615197703,
                    "subsample": 0.9395755830326895,
                    "reg_alpha": 3.9952613354787956,
                    "reg_lambda": 9.844890611010639,
                    "random_state": 724
                }
            },
            2022: {
                724: {
                    "n_estimators": 1267,
                    "max_depth": 9,
                    "num_leaves": 20,
                    "learning_rate": 0.2946384428484261,
                    "min_child_samples": 61,
                    "min_split_gain": 0.9817504089276279,
                    "colsample_bytree": 0.10793923832551586,
                    "subsample": 0.6017238780312595,
                    "reg_alpha": 9.958078172408374,
                    "reg_lambda": 0.16645744603399001,
                    "random_state": 724
                }
            }
        },
        "3순": {
            2018: {
                724: {
                    "n_estimators": 1453,
                    "max_depth": 20,
                    "num_leaves": 79,
                    "learning_rate": 0.29151227352036074,
                    "min_child_samples": 45,
                    "min_split_gain": 0.008889190142100545,
                    "colsample_bytree": 0.9923219213660308,
                    "subsample": 0.9947245470411638,
                    "reg_alpha": 6.00123641044576,
                    "reg_lambda": 0.07762752697423769,
                    "random_state": 724
                }
            },
            2019: {
                724: {
                    "n_estimators": 730,
                    "max_depth": 16,
                    "num_leaves": 109,
                    "learning_rate": 0.020762635802852064,
                    "min_child_samples": 11,
                    "min_split_gain": 0.3512962012777239,
                    "colsample_bytree": 0.818025947821749,
                    "subsample": 0.7576390892825418,
                    "reg_alpha": 6.680200128814104,
                    "reg_lambda": 7.315333241567051,
                    "random_state": 724
                }
            },
            2020: {
                724: {
                    "n_estimators": 886,
                    "max_depth": 8,
                    "num_leaves": 28,
                    "learning_rate": 0.09964089118003813,
                    "min_child_samples": 10,
                    "min_split_gain": 0.6865459287348828,
                    "colsample_bytree": 0.27661687733724083,
                    "subsample": 0.9031642154073226,
                    "reg_alpha": 3.434641793747079,
                    "reg_lambda": 4.627402112101245,
                    "random_state": 724
                }
            },
            2021: {
                724: {
                    "n_estimators": 1314,
                    "max_depth": 3,
                    "num_leaves": 77,
                    "learning_rate": 0.17039930994428143,
                    "min_child_samples": 14,
                    "min_split_gain": 0.37411969607618534,
                    "colsample_bytree": 0.618112374113434,
                    "subsample": 0.8556078217975212,
                    "reg_alpha": 2.585687130085454,
                    "reg_lambda": 7.985917975968532,
                    "random_state": 724
                }
            },
            2022: {
                724: {
                    "n_estimators": 1435,
                    "max_depth": 4,
                    "num_leaves": 37,
                    "learning_rate": 0.08035235302890018,
                    "min_child_samples": 64,
                    "min_split_gain": 0.467275703657525,
                    "colsample_bytree": 0.10567022890786815,
                    "subsample": 0.6746882948173147,
                    "reg_alpha": 4.683483045514258,
                    "reg_lambda": 0.0033548406931291908,
                    "random_state": 724
                }
            }
        }
    },
    "상추": {
        "1순": {
            2018: {
                724: {
                    "n_estimators": 1448,
                    "max_depth": 16,
                    "num_leaves": 51,
                    "learning_rate": 0.1827606550080943,
                    "min_child_samples": 23,
                    "min_split_gain": 0.6741232088006811,
                    "colsample_bytree": 0.22393428371293414,
                    "subsample": 0.8169448426912818,
                    "reg_alpha": 9.106436556274161,
                    "reg_lambda": 1.5115851375012406,
                    "random_state": 724
                }
            },
            2019: {
                724: {
                    "n_estimators": 1676,
                    "max_depth": 18,
                    "num_leaves": 114,
                    "learning_rate": 0.1630711944919535,
                    "min_child_samples": 10,
                    "min_split_gain": 0.24141809319960655,
                    "colsample_bytree": 0.11900503857196021,
                    "subsample": 0.5860599074265667,
                    "reg_alpha": 3.238379738535775,
                    "reg_lambda": 1.6663378655323857,
                    "random_state": 724
                }
            },
            2020: {
                724: {
                    "n_estimators": 533,
                    "max_depth": 10,
                    "num_leaves": 107,
                    "learning_rate": 0.1983783104150194,
                    "min_child_samples": 26,
                    "min_split_gain": 0.14790545346941575,
                    "colsample_bytree": 0.21973198421591783,
                    "subsample": 0.5213106423157854,
                    "reg_alpha": 2.824936458035757,
                    "reg_lambda": 4.798276572433707,
                    "random_state": 724
                }
            },
            2021: {
                724: {
                    "n_estimators": 2488,
                    "max_depth": 16,
                    "num_leaves": 143,
                    "learning_rate": 0.10189523564453562,
                    "min_child_samples": 21,
                    "min_split_gain": 0.037782280079819194,
                    "colsample_bytree": 0.2852420461146514,
                    "subsample": 0.5281348538671794,
                    "reg_alpha": 5.553093062516003,
                    "reg_lambda": 6.650427282184117,
                    "random_state": 724
                }
            },
            2022: {
                724: {
                    "n_estimators": 1056,
                    "max_depth": 14,
                    "num_leaves": 79,
                    "learning_rate": 0.10232330514884055,
                    "min_child_samples": 17,
                    "min_split_gain": 0.6680446980470558,
                    "colsample_bytree": 0.2697041808239538,
                    "subsample": 0.9657115512863422,
                    "reg_alpha": 9.624667220810542,
                    "reg_lambda": 7.895581104707648,
                    "random_state": 724
                }
            }
        },
        "2순": {
            2018: {
                724: {
                    "n_estimators": 1875,
                    "max_depth": 19,
                    "num_leaves": 44,
                    "learning_rate": 0.1401604796947311,
                    "min_child_samples": 8,
                    "min_split_gain": 0.5378424613525754,
                    "colsample_bytree": 0.18830131767519287,
                    "subsample": 0.7276071833461805,
                    "reg_alpha": 3.650076390197931,
                    "reg_lambda": 5.309770594135959,
                    "random_state": 724
                }
            },
            2019: {
                724: {
                    "n_estimators": 1211,
                    "max_depth": 12,
                    "num_leaves": 132,
                    "learning_rate": 0.06244975894504897,
                    "min_child_samples": 10,
                    "min_split_gain": 0.7594278939745809,
                    "colsample_bytree": 0.7050806526002555,
                    "subsample": 0.6895000560268629,
                    "reg_alpha": 2.5520759105143838,
                    "reg_lambda": 6.84076736995941,
                    "random_state": 724
                }
            },
            2020: {
                724: {
                    "n_estimators": 2979,
                    "max_depth": 3,
                    "num_leaves": 20,
                    "learning_rate": 0.018815297759352846,
                    "min_child_samples": 7,
                    "min_split_gain": 0.7845111253137251,
                    "colsample_bytree": 0.1052189950178265,
                    "subsample": 0.7510849142381776,
                    "reg_alpha": 9.373921478336976,
                    "reg_lambda": 0.3370219153219658,
                    "random_state": 724
                }
            },
            2021: {
                724: {
                    "n_estimators": 477,
                    "max_depth": 17,
                    "num_leaves": 84,
                    "learning_rate": 0.16135433052683965,
                    "min_child_samples": 5,
                    "min_split_gain": 0.609895811918732,
                    "colsample_bytree": 0.7771016476602417,
                    "subsample": 0.8535183057204787,
                    "reg_alpha": 5.874593714466089,
                    "reg_lambda": 3.4169063395707577,
                    "random_state": 724
                }
            },
            2022: {
                724: {
                    "n_estimators": 2446,
                    "max_depth": 3,
                    "num_leaves": 120,
                    "learning_rate": 0.11542260442609531,
                    "min_child_samples": 21,
                    "min_split_gain": 0.9790428696397763,
                    "colsample_bytree": 0.834506058826248,
                    "subsample": 0.9875977009625367,
                    "reg_alpha": 1.2197763570555213,
                    "reg_lambda": 4.177704932412051,
                    "random_state": 724
                }
            }
        },
        "3순": {
            2018: {
                724: {
                    "n_estimators": 933,
                    "max_depth": 18,
                    "num_leaves": 114,
                    "learning_rate": 0.06734463611866441,
                    "min_child_samples": 14,
                    "min_split_gain": 0.8735172580286712,
                    "colsample_bytree": 0.9225509412816814,
                    "subsample": 0.6815484948327433,
                    "reg_alpha": 5.648376842688621,
                    "reg_lambda": 1.4564146463376897,
                    "random_state": 724
                }
            },
            2019: {
                724: {
                    "n_estimators": 1151,
                    "max_depth": 16,
                    "num_leaves": 134,
                    "learning_rate": 0.03773505202698791,
                    "min_child_samples": 12,
                    "min_split_gain": 0.36550784161726,
                    "colsample_bytree": 0.6639042628382823,
                    "subsample": 0.992825428483524,
                    "reg_alpha": 8.639536306400778,
                    "reg_lambda": 6.4940653225552065,
                    "random_state": 724
                }
            },
            2020: {
                724: {
                    "n_estimators": 2333,
                    "max_depth": 11,
                    "num_leaves": 119,
                    "learning_rate": 0.11880699370738755,
                    "min_child_samples": 13,
                    "min_split_gain": 0.8744085136008527,
                    "colsample_bytree": 0.4160056694165536,
                    "subsample": 0.7464504012750607,
                    "reg_alpha": 1.2933567774098271,
                    "reg_lambda": 5.020250184617135,
                    "random_state": 724
                }
            },
            2021: {
                724: {
                    "n_estimators": 1823,
                    "max_depth": 14,
                    "num_leaves": 100,
                    "learning_rate": 0.11954273895853129,
                    "min_child_samples": 18,
                    "min_split_gain": 0.9973131325329946,
                    "colsample_bytree": 0.9200155660583661,
                    "subsample": 0.8551374137153488,
                    "reg_alpha": 6.229165705056072,
                    "reg_lambda": 1.015986636738701,
                    "random_state": 724
                }
            },
            2022: {
                724: {
                    "n_estimators": 324,
                    "max_depth": 10,
                    "num_leaves": 98,
                    "learning_rate": 0.28387552062370874,
                    "min_child_samples": 24,
                    "min_split_gain": 0.5064212014921596,
                    "colsample_bytree": 0.8162359212436944,
                    "subsample": 0.6418896244510196,
                    "reg_alpha": 4.887855911836028,
                    "reg_lambda": 0.12987875174992336,
                    "random_state": 724
                }
            }
        }
    }
}



for c in category:
    print(f'\n\n\n============================= {c} =============================')
    train = pd.read_csv(f"train_{c}.csv", encoding = 'cp949')
    test = pd.read_csv(f"test_{c}.csv", encoding = 'cp949')
    
    train, test = common_FE(train, test, c)
    y_preds = {}
    models = {}
    train_x = train.drop(['1순', '2순', '3순'], axis=1)
    train_y = train[['1순', '2순', '3순']].copy()
    test_original = test.drop(['1순', '2순', '3순'], axis=1)
    test_original.drop(['연도'],axis=1,inplace=True)

    test = test.drop(['1순', '2순', '3순'], axis=1)

    final_models = {}
    y_preds = {}
    seed_preds = []



    seeds = [724] 
    unique_years = [2018,2019,2020,2021,2022]

    for i in range(1, 4):  
        print(f'--------------------- {i}순 Optuna Optimization ---------------------')
                
        year_preds = []
        
        for year in unique_years:
            print(f'--------------------- Year {year} Excluded ---------------------')
            
            train_excluded = train[train['연도'] != year].copy()
            train_excluded.drop(['연도'],axis=1,inplace=True)
            train_excluded = train_excluded.reset_index(drop = True)
            X = train_excluded.drop(['1순', '2순', '3순'], axis=1)
            y = train_excluded[f'{i}순'].copy()

            seed_test_preds = []  

            for seed in seeds:
                print(f'--------------------- Seed {seed} ---------------------')
                
                test_x = test_original.copy()
                best_params = best_params_all[c][f'{i}순'][year][seed]
                best_params.pop('random_state')

                best_model = lgb.LGBMRegressor(**best_params, random_state = seed)
                print(best_model.get_params())
                best_model.fit(X, y)

                seed_test_preds.append(best_model.predict(test_x))

            year_pred = np.mean(seed_test_preds, axis=0)
            year_preds.append(year_pred)

        final_test_pred = np.mean(year_preds, axis=0)
        y_preds[f'y_pred_{i}'] = final_test_pred 
    
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
submission_data3.to_csv('머신러닝.csv',index=False,encoding='utf-8')

