# Spark Cluster with docker & docker-compose ('25)

## General

A simple spark standalone cluster for your testing environment purposes. A *docker compose up* away from your solution for your spark development environment.

The Docker compose will create the following containers:

container|Exposed ports
---|---
spark-master|9090 7077
spark-worker-a|9091
spark-worker-b|9092
demo-database|5432

## Installation

The following steps will make you run your spark cluster's containers.

### Prerequisites

* Docker installed

* Docker Compose  installed

### Build the image
This step will take some time as the spark package download is slow usually.

```sh
docker build -t cluster-apache-spark:3.3.4 .
```

### Run the docker-compose

The final step to create your test cluster will be to run the compose file.  Ensure that `docker-compose-v2` is already installed in the system.  If not, install via `sudo apt install docker-compose-v2`.

```sh
docker compose up -d
```

### Validate your cluster

Just validate your cluster by accessing the spark UI on each worker & master URL.

#### Spark Master

http://localhost:9090/

![alt text](articles/images/spark-master.png "Spark master UI")

#### Spark Worker 1

http://localhost:9091/

![alt text](articles/images/spark-worker-1.png "Spark worker 1 UI")

#### Spark Worker 2

http://localhost:9092/

![alt text](articles/images/spark-worker-2.png "Spark worker 2 UI")


## Resource Allocation 

This cluster is shipped with three workers and one spark master. Each of these has a particular resource allocation(basically, RAM and CPU core allocation).

* The default CPU core allocation for each spark worker is one core.

* The default RAM for each spark-worker is 1024 MB.

* The default RAM allocation for spark executors is 256 MB.

* The default RAM allocation for the spark driver is 128MB

* If you wish to modify this allocation, edit the env/spark-worker.sh file.

## Bind Volumes

To make the app running easier, I've shipped two volume mounts described in the following chart:

Host Mount|Container Mount|Purposse
---|---|---
apps|/opt/spark-apps|Used to make available your app's jars on all workers & master
data|/opt/spark-data| Used to make available your app's data on all workers & master

This is a dummy DFS created from docker Volumes...(maybe not...)

## Run Sample applications


### NY Bus Stops Data [Pyspark]

This program just loads archived data from [MTA Bus Time](http://web.mta.info/developers/MTA-Bus-Time-historical-data.html) and applies basic filters using Spark SQL. The results are persisted into a PostgreSQL table.

#### Download the dataset

Download the required dataset [MTA Bus Time](https://s3.amazonaws.com/nycbuspositions/2017/07/2017-07-14-bus-positions.csv.xz) by unzipping the data to **2017-07-14-bus-positions.csv** and place it under the **./data/** folder.

The loaded table will contain the following structure:

|trip_id|route_id|trip_start_time|trip_start_date|vehicle_id|vehicle_label|vehicle_license_plate|latitude|longitude|bearing|speed|stop_id|stop_status|occupancy_status|congestion_level|progress|block_assigned|dist_along_route|report_hour|report_date|
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|2017-07-14 00:15:24+00|CS_C7-Weekday-SDon-117400_MISC_825|Q28||13-07-2017|MTA|NYCT_7424|||40.765316|-73.816071|356.81|0|501000|IN_TRANSIT_TO|EMPTY|UNKNOWN_CONGESTION_LEVEL|||||2014-08-01 04:00:00|2014-08-01|

To submit the app, connect to one of the workers or the master and execute the following:
```sh
docker exec -it spark-master bash
```

```sh
/opt/spark/bin/spark-submit --master spark://spark-master:7077 \
--jars /opt/spark-apps/postgresql-42.2.22.jar \
--driver-memory 1G \
--executor-memory 1G \
/opt/spark-apps/main.py
```

![alt text](./articles/images/pyspark-demo.png "Spark UI with pyspark program running")

### MTA Bus Analytics [Scala]

This program takes the archived data from [MTA Bus Time](http://web.mta.info/developers/MTA-Bus-Time-historical-data.html) and makes some aggregations on it. The calculated results persist in Postgresql tables.

Each persisted table corresponds to a particular aggregation:

Table|Aggregation
---|---
day_summary|A summary of vehicles reporting, stops visited, average speed, and distance traveled(all vehicles)
speed_excesses|Speed excesses calculated in a 5-minute window
average_speed|Average speed by vehicle
distance_traveled|Total Distance traveled by vehicle


To submit the app connect to one of the workers or the master and execute:

```sh
/opt/spark/bin/spark-submit --deploy-mode cluster \
--master spark://spark-master:7077 \
--total-executor-cores 1 \
--class mta.processing.MTAStatisticsApp \
--driver-memory 1G \
--executor-memory 1G \
--jars /opt/spark-apps/postgresql-42.2.22.jar \
--conf spark.driver.extraJavaOptions='-Dconfig-path=/opt/spark-apps/mta.conf' \
--conf spark.executor.extraJavaOptions='-Dconfig-path=/opt/spark-apps/mta.conf' \
/opt/spark-apps/mta-processing.jar
```

You will notice on the spark-ui a driver program and executor program running(In scala we can use deploy-mode cluster)

![alt text](./articles/images/stats-app.png "Spark UI with scala program running")


## Summary

* We compiled the necessary docker image to run spark master and worker containers.

* We created a Spark standalone cluster using 2 worker nodes and 1 master node, using docker and docker-compose.

* Copied the resources necessary to run demo applications.

* We ran a distributed application at home(we just need enough CPU cores and RAM to do so).

## Why a standalone cluster?

* This is intended for test purposes. It is basically a way of running distributed Spark apps on your laptop or desktop.

* This will be useful to use CI/CD pipelines for your spark apps(A complicated and hot topic)

## Steps to connect and use a pyspark shell interactively

* Follow the steps to run the docker-compose file. If needed, you can scale this down to 1 worker. 

```sh
docker compose up --scale spark-worker=1
docker exec -it spark-worker_a bash
pyspark --master spark://spark-master:7077
```

## What's left to do?

* Right now, to run applications in the deploy-mode cluster, an arbitrary driver port must be specified.

* The spark-submit entry in the start-spark.sh is unimplemented; the submit used in the demos can be triggered by any worker
