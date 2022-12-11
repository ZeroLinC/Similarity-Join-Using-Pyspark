from pyspark import SparkContext, SparkConf
from math import ceil
import sys
import re
       
# yield token in prefix of each row
# in format (token, (rid, tokens))
# prefix_length = num_tokens - ceil(num_tokens*tau) + 1
# use frequency dictionary to create a sorted prefix list
# with the least freq token at beginning
# and most freq token at end
def prefix(line, tau, prefix_dict):
    values = line.split(" ")
    rid = values[0]
    if rid.isdigit:
        rid = int(rid)
    tokens = values[1:]
    prefix_length = len(tokens) - ceil(len(tokens)*tau) + 1
    sorted_prefix = []
    for token in tokens:
        new = [token, prefix_dict[token]]
        sorted_prefix.append(new)
    sorted_prefix = sorted(sorted_prefix, key = lambda x: (x[1],x[0]))
    for i in range(prefix_length):
        yield(sorted_prefix[i][0], (rid,tokens,len(tokens)))  

# filter the tokens pair with their length
# as for some pair, because of their length
# their similarity is always smaller than tau
# for len_y < len_x
# if len_y < len_x*tau
# their similarity is smaller than tau
def length_filter(size1, size2, tau):
    if size1 < size2:
        if size1/size2 >= tau:
            return True
        else:
            return False
    else:
        if size2/size1 >= tau:
            return True
        else:
            return False


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Wrong inputs")
        sys.exit(-1)
    else:
        file1 = sys.argv[1]
        file2 = sys.argv[2]
        tau = sys.argv[3]
        outputpath = sys.argv[4]
        
        # initial RDD
        conf = SparkConf().setAppName("Similarity_join")
        spark = SparkContext(conf=conf)
        
        # create RDD for file1 and file2
        f1 = spark.textFile(file1)
        f2 = spark.textFile(file2)
        
        # union two RDD in order to find token freq
        f1 = f1.flatMap(lambda line: line.strip().split("\n"))
        f2 = f2.flatMap(lambda line: line.strip().split("\n"))
        f = f1.union(f2)
        # find token freq like word_count
        tokens_frequency = f.flatMap(lambda line: list(set(line.split(" ")[1:])))\
                            .map(lambda token: (token, 1))\
                            .reduceByKey(lambda a,b : a+b)
                            
        # collect token freq as dictionary
        prefix_dict = {item[0]: item[1] for item in tokens_frequency.collect()}
        
        # find prefix token with (rid, tokens) for both file RDD
        prefixes_rdd1 = f1.flatMap(lambda line: prefix(line,float(tau),prefix_dict))
        prefixes_rdd2 = f2.flatMap(lambda line: prefix(line,float(tau),prefix_dict))
        
        # join them with same key
        # key is prefix token
        # if tokens of two pair has a similarity greater or equal to tau
        # they need at least one token in prefix the same 
        pairRDD = prefixes_rdd1.join(prefixes_rdd2)
        
        # length filter to reduce RDD
        pairRDD = pairRDD.filter(lambda x: length_filter(x[1][0][2],x[1][1][2],float(tau)))
        
        # check their similarity with tau
        res = pairRDD.map(lambda x: ((x[1][0][0],x[1][1][0]),(round((len(set(x[1][0][1]) & set(x[1][1][1]))/len(set(x[1][0][1]) | set(x[1][1][1]))),6))))\
                     .distinct()\
                     .filter(lambda x: x[1] >= float(tau))\
                     .sortBy(lambda x: (x[0][0],x[0][1]))
                     
        # output 
        res = res.map(lambda x: f"({x[0][0]},{x[0][1]})"+"\t"+f"{x[1]}").coalesce(1)
        res.saveAsTextFile(outputpath)
        
