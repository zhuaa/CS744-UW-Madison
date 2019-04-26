import sys
import re
from pyspark.sql import SparkSession
from operator import add

def computeContribs(urls, rank):
    """Calculates URL contributions to the rank of other URLs."""
    num_urls = len(urls)
    for url in urls:
        yield (url, rank / num_urls)


def parseNeighbors(urls):
    """Parses a urls pair string into urls pair."""
    parts = re.split(r'\t', urls, 1)
    parts[0] = parts[0].strip().lower()
    parts[1] = parts[1].strip().lower()
    #if (":" in parts[0] and parts[0].startswith("category:")==0)\
#	or (":" in parts[1] and parts[1].startswith("category:")==0):
 #     return "delete", "delete"
  #  else:
    return parts[0], parts[1]



spark = SparkSession\
	.builder\
	.appName("part3")\
	.config("spark.driver.memory", "8g")\
	.config("spark.executor.memory", "8g")\
	.config("spark.executor.cores", "5")\
	.config("spark.task.cpus", "1")\
	.getOrCreate()

file_num = 9
input_file_list = [sys.argv[i+3] for i in range(file_num)]
for i in range(len(input_file_list)):
	print(input_file_list[i])
	if i == 0:
		txt = spark.read.text(input_file_list[i])
	else:
		temp_txt = spark.read.text(input_file_list[i])
		txt = txt.union(temp_txt)

#txt = spark.read.text("hdfs://128.104.223.193:9000/part3/link-enwiki-20180601-pages-articles1.xml-p10p30302")
txt_rdd = txt.rdd.map(lambda r: r[0])

rows = txt_rdd.filter(lambda line: not ((':' in re.split(r'\t', line, 1)[0] and re.split(r'\t', line, 1)[0].startswith("Category:")==0) or (':' in re.split(r'\t', line, 1)[1] and re.split(r'\t', line, 1)[1].startswith("Category:")==0) or len(re.split(r'\t', line, 1))!=2))

links = rows.map(lambda urls: parseNeighbors(urls)).distinct().groupByKey().partitionBy(int(sys.argv[1])).cache()

ranks = links.map(lambda url_neighbors: (url_neighbors[0], 1.0)).partitionBy(int(sys.argv[1])).cache()


for ite in range(10):
	contributions = links.join(ranks).flatMap(
            lambda url_urls_rank: computeContribs(url_urls_rank[1][0], url_urls_rank[1][1]))
	ranks = contributions.reduceByKey(add).mapValues(lambda rank: rank * 0.85 + 0.15)

#df = spark.createDataFrame(ranks)
#df.write.format("csv").option("encoding", "UTF-8").save("/users/zihangm/part3_output/try7")
ranks.saveAsTextFile(sys.argv[2])
#df.write.save("/users/zihangm/part3_output/try9")
'''
ite = 0
for (link,rank) in ranks.collect():
	print("%s has rank: %s" %(link.encode("utf-8"),str(rank)))
	ite += 1
	if ite==5:
		break
'''
