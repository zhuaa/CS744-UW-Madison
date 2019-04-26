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
    parts = re.split(r'\s+', urls)
    return parts[0], parts[1]

spark = SparkSession\
	.builder\
	.appName("part3")\
	.config("spark.driver.memory", "8g")\
	.config("spark.executor.memory", "8g")\
	.config("spark.executor.cores", "5")\
	.config("spark.task.cpus", "1")\
	.getOrCreate()

txt = spark.read.text(sys.argv[2]).rdd.map(lambda r: r[0])
rows = txt.filter(lambda line: line[0]!="#")

links = rows.map(lambda urls: parseNeighbors(urls)).distinct().groupByKey().partitionBy(int(sys.argv[1])).cache()
ranks = links.map(lambda url_neighbors: (url_neighbors[0], 1.0)).partitionBy(int(sys.argv[1])).cache()

for ite in range(10):
	contributions = links.join(ranks).flatMap(
            lambda url_urls_rank: computeContribs(url_urls_rank[1][0], url_urls_rank[1][1]))
	ranks = contributions.reduceByKey(add).mapValues(lambda rank: rank * 0.85 + 0.15)

ite=0
for (link,rank) in ranks.collect():
	print("%s has rank: %s" %(link,rank))
	ite+=1
	if ite==5:
		break

