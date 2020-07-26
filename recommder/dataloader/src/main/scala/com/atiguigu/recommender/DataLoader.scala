package com.atiguigu.recommender

import com.mongodb.casbah.commons.MongoDBObject
import com.mongodb.casbah.{MongoClient, MongoClientURI}
import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
 *1                       #电影编号mid
 * Toy Story (1995)       #电影名称  name
 *                        #详情描述
 * 81 minutes              #时长  timelong
 * March 20, 2001         #发行时间   issue
 * 1995                   #拍摄时间    shoot
 * English                #语言        language
 * Adventure|Animation|Children|Comedy|Fantasy     #类型  genres
 * Tom Hanks|Tim Allen|Don Rickles|Jim Varney|Wallace Shawn|John Ratzenberger|Annie Potts|John Morris|Erik von Detten|Laurie Metcalf|R. Lee Ermey|Sarah Freeman|Penn Jillette|Tom Hanks|Tim Allen|Don Rickles|Jim Varney|Wallace Shawn
 * John Lasseter         #导演    directors
 * */
//非个性化的，基于统计的离线模块
//个性化的，基于隐语义协同过滤模型
//定义样例类进行包装
case class Movie(mid: Int, name: String, descri: String, timelong: String, issue: String,
                 shoot: String, language: String, genres: String, actors: String,
                 directors: String)
/*
* userId,movieId,rating,timestamp
     1,31,2.5,1260759144
* */
case class Rating(uid: Int, mid: Int, score: Double, timestamp: Int)
/*
* userId,movieId,tag,timestamp
    1,31,action,1260759144
* */
case class Tag(uid: Int, mid: Int, tag: String, timestamp: Int)

case class MongoConfig(uri:String, db:String)
case class ESConfig(httpHosts:String, transportHosts:String, index:String, clustername:String)

object DataLoader {
  //定义常量
  val MOVIE_DATA_PATH = "D:\\shangguigu\\tuijianxitong\\movierecomdersystem\\recommder\\dataloader\\src\\main\\resources\\movies.csv"
  val RATING_DATA_PATH = "D:\\shangguigu\\tuijianxitong\\movierecomdersystem\\recommder\\dataloader\\src\\main\\resources\\ratings.csv"
  val TAG_DATA_PATH = "D:\\shangguigu\\tuijianxitong\\movierecomdersystem\\recommder\\dataloader\\src\\main\\resources\\tags.csv"

  val MONGODB_MOVIE_COLLECTION = "Movie"
  val MONGODB_RATING_COLLECTION = "Rating"
  val MONGODB_TAG_COLLECTION = "Tag"
  val ES_MOVIE_INDEX = "Movie"

  def main(args: Array[String]): Unit = {
//    定义一些常量
    val config = Map(
      "spark.cores"->"local[*]",
      "mongo.uri"->"mongodb://localhost:27017/recommender",
      "mongo.db" -> "recommender",
      "es.httpHosts" -> "localhost:9200",
      "es.transportHosts" -> "localhost:9300",
      "es.index" -> "recommender",
      "es.cluster.name" -> "elasticsearch"
    )
    //创建一个sparkconf
    val sparkConf = new SparkConf().setMaster(config("spark.cores")).setAppName("DataLoader")
    //创建一个sparksession
    val spark = SparkSession.builder().config(sparkConf).getOrCreate()
    import spark.implicits._

    //加载数据
    val movieRDD = spark.sparkContext.textFile(MOVIE_DATA_PATH)
    val movieDF = movieRDD.map(
      item => {
        val attr = item.split("//^")
        Movie(attr(0).toInt,attr(1).trim,attr(2).trim,attr(3).trim,attr(4).trim,attr(5).trim,attr(6).trim,attr(7).trim,attr(8).trim,attr(9).trim)
      }
    ).toDF()


    val ratingRDD = spark.sparkContext.textFile(RATING_DATA_PATH)

    val ratingDF = ratingRDD.map(item => {
      val attr = item.split(",")
      Rating(attr(0).toInt,attr(1).toInt,attr(2).toInt,attr(3).toInt)
    }).toDF()

    val tagRDD = spark.sparkContext.textFile(TAG_DATA_PATH)
    //将tagRDD转换为DataFrame
    val tagDF = tagRDD.map(item => {
      val attr = item.split(",")
      Tag(attr(0).toInt,attr(1).toInt,attr(2).trim,attr(3).toInt)
    }).toDF()

    implicit val mongoconfig = MongoConfig(config("mongo.uri"),config("mongo.db"))

    //将数据保存到mongodb
    storeDataInMongoDB(movieDF,ratingDF,tagDF)

    //数据预处理，把movie对应的tag信息添加进去，加一列   tag1|tag2|tag3...
    import org.apache.spark.sql.functions._
    /*
    * mid,tags
    *
    * tags:tag1|tag2|tag3...
    * */
    val newTag = tagDF.groupBy($"mid")
      .agg(concat_ws("|",collect_set($"tag")).as("tags"))
      .select("mid","tags")

    //newTag和movie做join，数据合并在一起，默认使用inner 这里使用左外连接
    val movieWithTagsDF = movieDF.join(newTag,Seq("mid"),"left")
    implicit val esconfig = ESConfig(config("es.httpHosts"),config("es.transportHosts"),config("es.index"),config("es.cluster.name"))

  }

  def storeDataInMongoDB(movieDF: DataFrame, ratingDF: DataFrame, tagDF: DataFrame)(implicit mongoConfig: MongoConfig):Unit={
    //新建一个mongodb的连接
    val mongoClient = MongoClient(MongoClientURI(mongoConfig.uri))
    //如果mongodb中已经有相应的数据库，先删除
    mongoClient(mongoConfig.db)(MONGODB_MOVIE_COLLECTION).dropCollection()
    mongoClient(mongoConfig.db)(MONGODB_RATING_COLLECTION).dropCollection()
    mongoClient(mongoConfig.db)(MONGODB_TAG_COLLECTION).dropCollection()

    //将DF数据写入到对应的mongodb表中
    movieDF.write
      .option("uri",mongoConfig.uri)
      .option("collection",MONGODB_MOVIE_COLLECTION)
      .mode("overwrite")
      .format("com.mongodb.spark.sql")
      .save()
    tagDF.write
      .option("uri",mongoConfig.uri)
      .option("collection",MONGODB_TAG_COLLECTION)
      .mode("overwrite")
      .format("com.mongodb.spark.sql")
      .save()
    ratingDF.write
      .option("uri",mongoConfig.uri)
      .option("collection",MONGODB_RATING_COLLECTION)
      .mode("overwrite")
      .format("com.mongodb.spark.sql")
      .save()
    //对数据表建索引
    mongoClient(mongoConfig.db)(MONGODB_MOVIE_COLLECTION).createIndex(MongoDBObject("mid" -> 1))
    mongoClient(mongoConfig.db)(MONGODB_RATING_COLLECTION).createIndex(MongoDBObject("uid" -> 1))
    mongoClient(mongoConfig.db)(MONGODB_RATING_COLLECTION).createIndex(MongoDBObject("mid" -> 1))
    mongoClient(mongoConfig.db)(MONGODB_TAG_COLLECTION).createIndex(MongoDBObject("uid" -> 1))
    mongoClient(mongoConfig.db)(MONGODB_TAG_COLLECTION).createIndex(MongoDBObject("mid" -> 1))
    // 关闭MongoDB的连接
    mongoClient.close()

  }
















}
