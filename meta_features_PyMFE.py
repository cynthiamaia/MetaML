import sklearn.datasets
from pymfe.mfe import MFE
import warnings
import time
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import datetime
import logging


'''ids = 
[965,181,753,187,847,1511,979,1526,1509,1016,
1546,1545,1544,1543,1542,1541,1540,1539,1538,1537,1536,
1535,1534,1533,1532,1531,1530,1529,1528,1527,923,736,
719,994,1508,1507,788,885,1506,1115,955,1004,770,841,
1503,953,737,902,1502,826,958,1498,1520,717,1496,816,
752,996,722,155,1567,354,750,1490,1019,1069,1167,1488,1021,980,959,
886,784,881,880,164,995,962,1020,971,1056,733,10,941,977,1483,184,1481,807,1045,1066,1048,1073,
969,823,843,821,1513,53,1565,1026,1005,714,901,805,838,855,918,933,863,932,868,
878,797,723,751,937,749,896,936,873,744,832,793,716,916,768,783,806,813,
715,740,920,792,879,869,877,911,794,830,922,726,775,762,866,912,903,913,766,
870,779,824,769,730,746,935,876,829,812,789,837,743,917,910,888,884,926,943,732,776,
773,763,850,754,889,808,904,799,849,845,1012,1473,1044,1011,931,795,827,774,818,819,803,1075,735,761,796,150,987,351,983,890,900,906,907,908,909,1560,991,23499,825,853,778,1463,725,833,997,1121,463,745,1553,1554,1552,1549,1551,1555,756,951,947,950,949,1061,1059,748,724,728,921,
771,450,1025,1014,875,444,448,
970,767,734,1556,720,337,336,160,159,158,157,156,1100,1444,1453,1452,1442,1113,976,1446,1447,272]'''
# ids = [271,141,268]
#264,130,261,125,259,123,254,252,250,249,74,72,146,267,265,119,727]'''
ids = [976,727]

try:
  df_array=pd.DataFrame()
  array_dados=[]
  array_columns=[]
  array_tempo=[]
  array_ids=[]
  dicionario = {}

  hora_file = datetime.datetime.now()
  hora_file_time_str = hora_file.strftime("%d-%m-%Y_%H-%M-%S")
  nome_file_log = "logfilename_"+ hora_file_time_str  + ".log"

  # logging.basicConfig(filename=nome_file_log, level=logging.INFO)

  logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename=nome_file_log,
                    filemode='w')
  console = logging.StreamHandler()
  console.setLevel(logging.INFO)
  # add the handler to the root logger
  logging.getLogger('').addHandler(console)


  for i in range(len(ids)):
    inicio = time.time()
    logging.info(f"start of time, {inicio}")
    logging.info(f"entered the for. dataset: {ids[i]}")
    X, y = sklearn.datasets.fetch_openml(data_id=ids[i], return_X_y=True,as_frame=False)
    if type(X) == np.ndarray:
      XX = X
    else:
      XX = X.toarray()
    #mfe = MFE(groups=["general", "statistical", "info-theory","complexity","landmarking","model-based"])
    # import ipdb
    # ipdb.set_trace()
    logging.info(f"START MFE = {ids[i]}")
    mfe = MFE(groups=["complexity"])
    #mfe = MFE(groups=["model-based"], features=["leaves","var_importance","tree_imbalance","leaves_branch","nodes_per_inst"])
    logging.info(f"FIT MFE. = {ids[i]}")
    mfe.fit(XX, y)
    logging.info(f"Extraction MFE = {ids[i]}")
    ft = mfe.extract()
    if not array_columns:
      # array_columns=['ids'] + ft[0]
      array_columns=ft[0]
    logging.info(f"Extraction MFE = {ids[i]}")
    array_dados.append(ft[1])
    array_ids.append(ids[i])
    logging.info(f"Output for = {array_dados}")
    fim = time.time()
    logging.info(f"End of time, {fim}")
    tempo_total = fim-inicio
    dicionario = {"id_dataset": ids[i], "total time": tempo_total}
    array_tempo.append(dicionario)


    logging.info(f"total time = {tempo_total} id dataset: {ids[i]}")
    logging.info(f"Weather dictionary based on{array_tempo}")
    logging.info(f"metric data {array_dados}")

  df = pd.DataFrame(array_tempo)
  # array_ids
  df_ids = pd.DataFrame(array_ids, columns=['ids'])
  logging.info(f"RESULTADO DOS IDS EXTRAIDOS: {df_ids}")
  df_dados = pd.DataFrame(array_dados, columns=array_columns)
  logging.info(f"RESULTADO DO DATAFRAME DAS METAS CARACTERISCAS: {df_dados}")
  df_resultado_total = pd.concat([df_ids, df_dados], axis=1)
  logging.info(df)
  data_file = datetime.datetime.now()
  date_time_str = data_file.strftime("%d-%m-%Y_%H-%M-%S")
  nome_file = "C:\\project-metafeatures\\metafeatures\\complexity\\information_dates_time"+ date_time_str  + ".csv"
  nome_file_bases = "C:\\project-metafeatures\\metafeatures\\complexity\\information_dates"+ date_time_str  + ".csv"

  df.to_csv(nome_file, sep = ';', index = False)
  df_resultado_total.to_csv(nome_file_bases, sep = ';', index = False)
except Exception as e:
  logging.error(f"Error {e}")