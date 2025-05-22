from pyspark.sql import SparkSession
from pyspark.sql.functions import col,date_format

# download the data from https://s3.amazonaws.com/nycbuspositions/2017/07/2017-07-14-bus-positions.csv.xz
# and unzip it to as ./data/2017-07-14-bus-positions.csv

def init_spark():
  sql = SparkSession.builder\
    .appName("trip-app")\
    .config("spark.jars", "/opt/spark-apps/postgresql-42.2.22.jar")\
    .getOrCreate()
  sc = sql.sparkContext
  return sql,sc

def main():
  url = "jdbc:postgresql://demo-database:5432/postgres"
  properties = {
    "user": "postgres",
    "password": "casa1234",
    "driver": "org.postgresql.Driver"
  }
  file = "/opt/spark-data/2017-07-14-bus-positions.csv"
  sql,sc = init_spark()

  df = sql.read.load(file,format = "csv", inferSchema="true", sep=",", header="true") \
      .withColumn("report_hour",date_format(col("timestamp"),"yyyy-MM-dd HH:00:00")) \
      .withColumn("report_date",date_format(col("timestamp"),"yyyy-MM-dd"))
  
  # Filter invalid coordinates and save the results to public/mta_reports table.
  df.where("latitude <= 90 AND latitude >= -90 AND longitude <= 180 AND longitude >= -180") \
    .where("latitude != 0.000000 OR longitude !=  0.000000 ") \
    .write \
    .jdbc(url=url, table="mta_data.mta_reports", mode='append', properties=properties)  
    # instead of putting the output at public.mta_reports, let push it to mta_data.mta_reports

if __name__ == '__main__':
  main()
