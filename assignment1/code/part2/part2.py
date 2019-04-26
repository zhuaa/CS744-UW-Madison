import sys
from pyspark.sql import SparkSession

spark = SparkSession\
	.builder\
	.appName("part2_sort")\
	.config("spark.driver.memory", "8g")\
	.config("spark.executer.memory", "8g")\
	.config("spark.task.cpus", "1")\
	.config("spark.executor.cores", "5")\
	.getOrCreate()

input_path = sys.argv[1]
output_path = sys.argv[2]

csv_file = spark.read.format("csv").load(input_path).rdd
#temp = csv_file.toDF()
header = csv_file.first()
rows = csv_file.filter(lambda line:line!=header)
### map: get ((key1, key2), values)
pairs = rows.map(lambda row: ((row[2], row[14]), row))

result = pairs.sortByKey()
values = result.map(lambda r:r[1])


df = spark.createDataFrame(values, header)

df.write.format("csv").option("header", "true").save(output_path)

#values.write.csv(output_path, "append")
	#.format("csv")
	#.mode("append")
	#.save(output_path)

#print(result.take(10))
#print("finish!!!!!!!!!!!!!!!")
