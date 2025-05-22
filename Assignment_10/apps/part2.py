from pyspark.sql import SparkSession
import numpy as np
import time
import matplotlib.pyplot as plt

def init_spark(app_name="VectorOps"):
    return SparkSession.builder.appName(app_name).getOrCreate()

def vector_dot_product(spark, arr1, arr2, num_slices: int = 4):
    parallelized_arr1 = spark.sparkContext.parallelize(arr1, num_slices)
    parallelized_arr2 = spark.sparkContext.parallelize(arr2, num_slices)
    return parallelized_arr1.zip(parallelized_arr2).map(lambda x: x[0] * x[1]).reduce(lambda x, y: x + y)

def vector_scaling(spark, arr1, scale: int = 5, num_slices: int = 4):
    parallelized_arr1 = spark.sparkContext.parallelize(arr1, num_slices)
    return parallelized_arr1.map(lambda x: scale * x).collect()

def vector_addition(spark, arr1, arr2, num_slices: int = 4):
    parallelized_arr1 = spark.sparkContext.parallelize(arr1, num_slices)
    parallelized_arr2 = spark.sparkContext.parallelize(arr2, num_slices)
    return parallelized_arr1.zip(parallelized_arr2).map(lambda x: x[0] + x[1]).collect()

if __name__ == "__main__":
    spark = init_spark()

    array_sizes = [int(1e5), int(1e6), int(3*1e7)]
    num_slices_arr = [2, 12, 100]

    times = {a_size : [] for a_size in array_sizes}

    for a_size in array_sizes:
        for n_slice in num_slices_arr:
            arr1 = np.random.rand(a_size)
            arr2 = np.random.rand(a_size)

            start_time = time.time()
            vector_dot_product(spark, arr1, arr2, n_slice)

            vector_addition(spark, arr1, arr2, n_slice)

            vector_scaling(spark, arr1, 5, n_slice)
            end_time = time.time() - start_time

            times[a_size].append(end_time)

    # Plotting
    plt.figure(figsize=(10, 6))
    for a_size in array_sizes:
        plt.loglog(num_slices_arr, times[a_size], marker='o', label=f"Array size {a_size}")
    plt.title(f"Time vs Number of Slices")
    plt.xlabel("Number of Slices")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'final.png')
    plt.show()