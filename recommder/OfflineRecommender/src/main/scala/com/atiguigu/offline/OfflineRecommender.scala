package com.atiguigu.offline

import org.apache.spark.SparkConf

import org.apache.spark.sql.SparkSession
import org.jblas.DoubleMatrix

import org.apache.spark.mllib.recommendation.{ALS, Rating}
//基于评分数据的LFM（隐语义模型），只需要rating数据
case class MovieRating(uid: Int, mid: Int, score: Double, timestamp: Int)
case class MongoConfig(uri:String, db:String)
//定义一个基准推荐对象
case class Recommendation(mid:Int,score:Double)
//定义基于预测评分的用户推荐列表
case class UserRecs(uid:Int,recs:Seq[Recommendation])
//定义基于LFM电影特征向量的电影相似度列表
case class MovieRecs(mid:Int,recs:Seq[Recommendation])

object OfflineRecommender {
  //定义表名和常量
  val MONGODB_RATING_COLLECTION = "Rating"
  val USER_RECS = "UserRecs"
  val MOVIE_RECS = "MovieRecs"
  val USER_MAX_RECOMMENDATION = 20
  val RATING_DATA_PATH = "D:\\shangguigu\\tuijianxitong\\movierecomdersystem\\recommder\\dataloader\\src\\main\\resources\\ratings.csv"


  def main(args: Array[String]): Unit = {
    val config = Map(
      "spark.cores"->"local[*]",
      "mongo.uri"->"mongodb://localhost:27017/recommender",
      "mongo.db" -> "recommender"
    )
    val sparkConf = new SparkConf().setMaster(config("spark.cores")).setAppName("OfflineRecommender")
    //创建一个SparkSession
    val spark = SparkSession.builder().config(sparkConf).getOrCreate()
    implicit val mongoConfig = MongoConfig(config("mongo.uri"),config("mongo.db"))
    import spark.implicits._
    //加载数据
//    val ratingRDD = spark.read
//      .option("uri",mongoConfig.uri)
//      .option("collection",MONGODB_RATING_COLLECTION)
//      .format("com.mongodb.spark.sql")
//      .load()
//      .as[MovieRating]
//      .rdd
//      .map(rating =>(rating.uid,rating.score))
//      .toDS()
//      .as[MovieRating]
//      .rdd
//      .map( rating => (rating.uid,rating.mid,rating.score))
//      .cache()
    val ratingRDD1 = spark.sparkContext.textFile(RATING_DATA_PATH)
    //直接从文件里面读出来的就是rdd，从关系型数据库里面得到的是dataframe
    val ratingRDD = ratingRDD1.map(
      items =>{
        val attr = items.split(",")
        MovieRating(attr(0).toInt,attr(1).toInt,attr(2).toDouble,attr(3).toInt)
      }
    )
      .map(rating=>(rating.uid,rating.mid,rating.score))
      .cache()
    //从rating数据中提取所有的uid和mid并去重
    val userRDD = ratingRDD.map(_._1).distinct()
    val movieRDD = ratingRDD.map(_._2).distinct()
    //训练隐语义模型
    val trainData = ratingRDD.map(x => Rating(x._1,x._2,x._3))
    val (rank,iterations,lambda) = (50,5,0.01)
    val model = ALS.train(trainData,rank,iterations,lambda)
    //基于用户和电影的隐特征，计算预测评分，得到用户的推荐列表
    //计算user和movie的笛卡尔积，得到一个空评分矩阵
    val userMovies = userRDD.cartesian(movieRDD)
    //调用model的predict方法预测评分,返回值是RDD[Rating]，这个Rating是mllib里面定义的
    val preRatings = model.predict(userMovies)
    val userRecs = preRatings
      .filter(_.rating > 0)   //preRatings里面是Rating，里面有三列，rating是最后一列
      .map(rating => (rating.user,(rating.product,rating.rating)))
      .groupByKey()
      .map{
        case (uid,recs) => UserRecs(uid,recs.toList.sortWith(_._2>_._2).take(20).map(x=>Recommendation(x._1,x._2)))
      }
      .toDF()
//    userRecs.show(false)
    //基于电影隐特征，计算相似度矩阵，得到电影的相似度列表
    val movieFeature = model.productFeatures.map{
      case (mid,features) => (mid,new DoubleMatrix(features))
    }
    //对所有电影两两计算它们的相似度，先做笛卡尔积
    val movieRecs = movieFeature.cartesian(movieFeature)
      .filter{
        //把自己跟自己的配对过滤掉
        case (a,b) => a._1 != b._1
      }
      .map{
        case (a,b) =>{
          val simScore = this.consinSim(a._2,b._2)
          (a._1,(b._1,simScore))
        }
      }
      .filter(_._2._2 >= 0.6)    //过滤出相似度大于0.6的
      .groupByKey()
      .map{
        case (mid,items) => MovieRecs(mid,items.toList.sortWith(_._2>_._2).map(x=>Recommendation(x._1,x._2)))
      }
        .toDF()
    movieRecs.show(false)
    spark.stop()
  }
  //向量余弦相似度
  def consinSim(movie1: DoubleMatrix, movie2: DoubleMatrix):Double={
    movie1.dot(movie2)/(movie1.norm2()*movie2.norm2())
  }


}
