m1 = pd.read_csv('머신러닝.csv')
m2 = pd.read_csv('시계열회귀.csv')

ml = m1.copy()
ml.iloc[:,1:] = (m1.iloc[:,1:]*0.5 + m2.iloc[:,1:]*0.5)

d1 = pd.read_csv('dl_model1.csv')
d2 = pd.read_csv('dl_model2.csv')

dl = d2.copy()

dl.loc[dl['시점'].str.endswith('+1순'), dl.columns != '시점'] = (0.5 * d2.loc[dl['시점'].str.endswith('+1순'), dl.columns != '시점'] + 0.5 * d1.loc[dl['시점'].str.endswith('+1순'), dl.columns != '시점'])

submit = pd.read_csv('sample_submission.csv')
submit.loc[submit['시점'].str.endswith('+1순'), '감자 수미'] = 1 * ml.loc[submit['시점'].str.endswith('+1순'), '감자 수미']
submit.loc[submit['시점'].str.endswith('+1순'), '건고추'] = 0.1 * ml.loc[submit['시점'].str.endswith('+1순'), '건고추'] + 0.9 * dl.loc[submit['시점'].str.endswith('+1순'), '건고추']
submit.loc[submit['시점'].str.endswith('+1순'), '깐마늘(국산)'] = 0.1 * ml.loc[submit['시점'].str.endswith('+1순'), '깐마늘(국산)'] + 0.9 * dl.loc[submit['시점'].str.endswith('+1순'), '깐마늘(국산)']
submit.loc[submit['시점'].str.endswith('+1순'), '대파(일반)'] = 0.7 * ml.loc[submit['시점'].str.endswith('+1순'), '대파(일반)'] + 0.3 * dl.loc[submit['시점'].str.endswith('+1순'), '대파(일반)']
submit.loc[submit['시점'].str.endswith('+1순'), '무'] = 0.5 * ml.loc[submit['시점'].str.endswith('+1순'), '무'] + 0.5 * dl.loc[submit['시점'].str.endswith('+1순'), '무']
submit.loc[submit['시점'].str.endswith('+1순'), '배추'] = 1 * ml.loc[submit['시점'].str.endswith('+1순'), '배추']
submit.loc[submit['시점'].str.endswith('+1순'), '사과'] = 0.5 * ml.loc[submit['시점'].str.endswith('+1순'), '사과'] + 0.5 * dl.loc[submit['시점'].str.endswith('+1순'), '사과']
submit.loc[submit['시점'].str.endswith('+1순'), '상추'] = 0.5 * ml.loc[submit['시점'].str.endswith('+1순'), '상추'] + 0.5 * dl.loc[submit['시점'].str.endswith('+1순'), '상추']
submit.loc[submit['시점'].str.endswith('+1순'), '양파'] = 0.5 * ml.loc[submit['시점'].str.endswith('+1순'), '양파'] + 0.5 * dl.loc[submit['시점'].str.endswith('+1순'), '양파']
submit.loc[submit['시점'].str.endswith('+1순'), '배'] = 1 * dl.loc[submit['시점'].str.endswith('+1순'), '배']

submit.loc[submit['시점'].str.endswith('+2순'), '감자 수미'] = 1 * ml.loc[submit['시점'].str.endswith('+2순'), '감자 수미']
submit.loc[submit['시점'].str.endswith('+2순'), '건고추'] = 0.3 * ml.loc[submit['시점'].str.endswith('+2순'), '건고추'] + 0.7 * dl.loc[submit['시점'].str.endswith('+2순'), '건고추']
submit.loc[submit['시점'].str.endswith('+2순'), '깐마늘(국산)'] = 0.3 * ml.loc[submit['시점'].str.endswith('+2순'), '깐마늘(국산)'] + 0.7 * dl.loc[submit['시점'].str.endswith('+2순'), '깐마늘(국산)']
submit.loc[submit['시점'].str.endswith('+2순'), '대파(일반)'] = 0.5 * ml.loc[submit['시점'].str.endswith('+2순'), '대파(일반)'] + 0.5 * dl.loc[submit['시점'].str.endswith('+2순'), '대파(일반)']
submit.loc[submit['시점'].str.endswith('+2순'), '무'] = 0.9 * ml.loc[submit['시점'].str.endswith('+2순'), '무'] + 0.1 * dl.loc[submit['시점'].str.endswith('+2순'), '무']
submit.loc[submit['시점'].str.endswith('+2순'), '배추'] = 1 * ml.loc[submit['시점'].str.endswith('+2순'), '배추']
submit.loc[submit['시점'].str.endswith('+2순'), '사과'] = 0.9 * ml.loc[submit['시점'].str.endswith('+2순'), '사과'] + 0.1 * dl.loc[submit['시점'].str.endswith('+2순'), '사과']
submit.loc[submit['시점'].str.endswith('+2순'), '상추'] = 0.7 * ml.loc[submit['시점'].str.endswith('+2순'), '상추'] + 0.3 * dl.loc[submit['시점'].str.endswith('+2순'), '상추']
submit.loc[submit['시점'].str.endswith('+2순'), '양파'] = 0.7 * ml.loc[submit['시점'].str.endswith('+2순'), '양파'] + 0.3 * dl.loc[submit['시점'].str.endswith('+2순'), '양파']
submit.loc[submit['시점'].str.endswith('+2순'), '배'] = 1 * dl.loc[submit['시점'].str.endswith('+2순'), '배']

submit.loc[submit['시점'].str.endswith('+3순'), '감자 수미'] = 1 * ml.loc[submit['시점'].str.endswith('+3순'), '감자 수미']
submit.loc[submit['시점'].str.endswith('+3순'), '건고추'] = 0.1 * ml.loc[submit['시점'].str.endswith('+3순'), '건고추'] + 0.9 * dl.loc[submit['시점'].str.endswith('+3순'), '건고추']
submit.loc[submit['시점'].str.endswith('+3순'), '깐마늘(국산)'] = 0.5 * ml.loc[submit['시점'].str.endswith('+3순'), '깐마늘(국산)'] + 0.5 * dl.loc[submit['시점'].str.endswith('+3순'), '깐마늘(국산)']
submit.loc[submit['시점'].str.endswith('+3순'), '대파(일반)'] = 0.5 * ml.loc[submit['시점'].str.endswith('+3순'), '대파(일반)'] + 0.5 * dl.loc[submit['시점'].str.endswith('+3순'), '대파(일반)']
submit.loc[submit['시점'].str.endswith('+3순'), '무'] = 0.9 * ml.loc[submit['시점'].str.endswith('+3순'), '무'] + 0.1 * dl.loc[submit['시점'].str.endswith('+3순'), '무']
submit.loc[submit['시점'].str.endswith('+3순'), '배추'] = 1 * ml.loc[submit['시점'].str.endswith('+3순'), '배추']
submit.loc[submit['시점'].str.endswith('+3순'), '사과'] = 0.9 * ml.loc[submit['시점'].str.endswith('+3순'), '사과'] + 0.1 * dl.loc[submit['시점'].str.endswith('+3순'), '사과']
submit.loc[submit['시점'].str.endswith('+3순'), '상추'] = 0.9 * ml.loc[submit['시점'].str.endswith('+3순'), '상추'] + 0.1 * dl.loc[submit['시점'].str.endswith('+3순'), '상추']
submit.loc[submit['시점'].str.endswith('+3순'), '양파'] = 0.5 * ml.loc[submit['시점'].str.endswith('+3순'), '양파'] + 0.5 * dl.loc[submit['시점'].str.endswith('+3순'), '양파']
submit.loc[submit['시점'].str.endswith('+3순'), '배'] = 1 * dl.loc[submit['시점'].str.endswith('+3순'), '배']


columns_to_round = ['건고추', '배', '사과', '상추','깐마늘(국산)']
submit[columns_to_round] = submit[columns_to_round].round()

submit.to_csv('Final.csv',index=False)
