import logging
# 先创建一个logger
logger = logging.getLogger(__name__)  # 定义Logger的名字，之前直接用logging调用的名字是root，日志格式用%(name)s可以获得。这里的名字也可以自定义比如"TEST"
logger.setLevel(logging.DEBUG)  # 低于这个级别将被忽略，后面还可以设置输出级别
# 创建handler和输出级别
ch = logging.StreamHandler()  # 输出到屏幕的handler
ch.setLevel(logging.INFO)  # 输出级别和上面的忽略级别都不一样，可以看一下效果
fh = logging.FileHandler('access.log',encoding='utf-8')  # 输出到文件的handler，定义一下字符编码
fh.setLevel(logging.WARNING)
# 创建日志格式，可以为每个handler创建不同的格式
ch_formatter = logging.Formatter('%(name)s %(asctime)s {%(levelname)s}:%(message)s',datefmt='%Y-%m-%d %H:%M:%S')  # 关键参数datefmt自定义日期格式
fh_formatter = logging.Formatter('%(asctime)s %(module)s-%(lineno)d [%(levelname)s]:%(message)s',datefmt='%Y/%m/%d %H:%M:%S')
# 把上面的日志格式和handler关联起来
ch.setFormatter(ch_formatter)
fh.setFormatter(fh_formatter)
# 将handler加入logger
logger.addHandler(ch)
logger.addHandler(fh)
# 以上就完成了，下面来看一下输出的日志
logger.debug('logger test debug')
logger.info('logger test info')
logger.warning('logger test warning')
logger.error('logger test error')
logger.critical('logger test critical')