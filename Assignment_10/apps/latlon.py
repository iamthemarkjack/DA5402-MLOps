from pyspark import SparkContext
import pandas as pd
from operator import add

def tuple_add(a, b):
    return (a[0]+b[0], a[1]+b[1])

def main():
    sc = SparkContext(master="spark://spark-master:7077")

    df = pd.read_csv('/opt/spark-data/2017-07-14-bus-positions.csv')
    data = df[['route_id', 'latitude', 'longitude']].values
    
    rdd = sc.parallelize(data, 20)
    #rdd.cache()

    # get the latitudes & longitudes
    lat_lon = rdd.map(lambda x: (x[0], (x[1], x[2]))).reduceByKey(tuple_add)

    # compute the lengths
    lengths = rdd.groupBy(lambda x: x[0]).mapValues(lambda x: len(x))

    # join the rdds
    merged = lat_lon.join(lengths)

    # compute the average lat lon values.
    final = merged.map(lambda x: (x[0], (x[1][0][0]/x[1][1], x[1][0][1]/x[1][1])))

    results = final.collect()

    with open("/opt/spark-data/results", "w") as file:
       for result in results:
          file.write(result[0] + " -> " + str(result[1]) + "\n")
       file.close()

if __name__ == '__main__':
  main()


