package com.atiguigu.offline

import breeze.numerics.sqrt
import com.atiguigu.offline.OfflineRecommender.MONGODB_RATING_COLLECTION
import org.apache.spark.SparkConf
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

object ALSTrain {
  def main(args: Array[String]): Unit = {
    val RATING_DATA_PATH = "D:\\shangguigu\\tuijianxitong\\movierecomdersystem\\recommder\\dataloader\\src\\main\\resources\\ratings.csv"
    val config = Map(
      "spark.cores"->"local[*]",
      "mongo.uri"->"mongodb://localhost:27017/recommender",
      "mongo.db" -> "recommender"
    )
    val sparkConf = new SparkConf().setMaster(config("spark.cores")).setAppName("ALSTrain")
    //创建一个SparkSession
    val spark = SparkSession.builder().config(sparkConf).getOrCreate()
    import spark.implicits._
//    implicit val mongoConfig = MongoConfig(config("mongo.uri"),config("mongo.db"))
    //加载数据
    val ratingRDD1 = spark.sparkContext.textFile(RATING_DATA_PATH)
    //直接从文件里面读出来的就是rdd，从关系型数据库里面得到的是dataframe
    val ratingRDD = ratingRDD1.map(
      items =>{
        val attr = items.split(",")
        MovieRating(attr(0).toInt,attr(1).toInt,attr(2).toDouble,attr(3).toInt)
      }
    )
      .map(rating=>Rating(rating.uid,rating.mid,rating.score))
      .cache()
    /**
     *

    val ratingRDD = spark.read
      .option("uri",mongoConfig.uri)
      .option("collection",MONGODB_RATING_COLLECTION)
      .format("com.mongodb.spark.sql")
      .load()
      .as[MovieRating]
      .rdd
      .map( rating => Rating(rating.uid,rating.mid,rating.score))
      .cache()
     */
    //随机切分数据集，生成训练集和测试集
    val splits = ratingRDD.randomSplit(Array(0.8,0.2))
    val trainRDD = splits(0)
    val testRDD = splits(1)
    //模型参数选择，输出最优参数
    adjustALSParam(trainRDD,testRDD)
    spark.close()
  }
  def adjustALSParam(trainData: RDD[Rating], testData: RDD[Rating]): Unit ={
    val result = for(rank <- Array(20,50,100);lambda <- Array(0.001,0.01,0.1))
      yield {
        val model = ALS.train(trainData,rank,5,lambda)
        //计算当前参数对应的模型的rmse，返回Double
        val rmse = getRMSE(model,testData)
        (rank,lambda,rmse)
      }
    //控制台打印输出最优参数
    println(result.minBy(_._3))
  }
  def getRMSE(model: MatrixFactorizationModel, data: RDD[Rating]):Double={
    //计算预测评分
    val userProducts = data.map(item => (item.user,item.product))
    val predictRating = model.predict(userProducts)

    //以uid,mid作为外键，inner join实际观测值和预测值
    val observed = data.map(item => ((item.user,item.product),item.rating))
    val predict = predictRating.map(item => ((item.user,item.product),item.rating))

      //内连接得到（uid,mid),(actual,pre)
    sqrt(
      observed.join(predict).map{
        case ((uid,mid),(actual,pre)) =>
          val err = actual - pre
          err*err
      }.mean()
    )

  }

}
















