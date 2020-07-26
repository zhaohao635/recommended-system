package com.atiguigu.content

import org.apache.spark.SparkConf
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.sql.SparkSession
import org.jblas.DoubleMatrix

case class Movie(mid: Int, name: String, descri: String, timelong: String, issue: String,
                 shoot: String, language: String, genres: String, actors: String,
                 directors: String)

case class MongoConfig(uri:String, db:String)
//定义一个基准推荐对象
case class Recommendation(mid:Int,score:Double)
//定义基于LFM电影特征向量的电影相似度列表
case class MovieRecs(mid:Int,recs:Seq[Recommendation])

object ContentRecommender1 {
  val MONGODB_MOVIE_COLLECTION = "Movie"
  val CONTENT_MOVIE_RECS = "ContentMovieRecs"
  val MOVIE_RECS = "MovieRecs"
  val MOVIE_DATA_PATH = "D:\\shangguigu\\tuijianxitong\\movierecomdersystem\\recommder\\dataloader\\src\\main\\resources\\movies.csv"

  def main(args: Array[String]): Unit = {
    val config = Map(
      "spark.cores"->"local[*]",
      "mongo.uri"->"mongodb://localhost:27017/recommender",
      "mongo.db" -> "recommender"
    )
    val sparkConf = new SparkConf().setMaster(config("spark.cores")).setAppName("contentRecommender")
    val spark = SparkSession.builder().config(sparkConf).getOrCreate()
    import spark.implicits._
    implicit val mongoConfig = MongoConfig(config("mongo.uri"),config("mongo.db"))

    //加载数据，并做预处理
    val movieRDD = spark.sparkContext.textFile(MOVIE_DATA_PATH)
    val movieDF = movieRDD.map(
      item => {
        val attr = item.split("\\^")
        Movie(attr(0).toInt,attr(1).trim,attr(2).trim,attr(3).trim,attr(4).trim,attr(5).trim,attr(6).trim,attr(7).trim,attr(8).trim,attr(9).trim)
      }
    ).map(
      x=>(x.mid,x.name,x.genres.map(c=>if(c=='|') ' ' else c))
    )
      .toDF("mid","name","genres")
      .cache()
//    val movieTagsDF = spark.read
//      .option("uri",mongoConfig.uri)
//      .option("collection",MONGODB_MOVIE_COLLECTION)
//      .format("com.mongodb.spark.sql")
//      .load()
//      .as[Movie]
//      .map(
//        //提取mid,name,genres三项作为原始内容特征，分词器默认按照空格做分词
//        x => (x.mid,x.name,x.genres.map(c => if(c=='|') ' ' else c))
//      )
//      .toDF("mid","name","genres")
//      .cache()
    //核心部分，用TF-IDF从内容信息中提取电影特征向量
    //创建一个分词器，默认按照空格分词
    val tokenizer = new Tokenizer().setInputCol("genres").setOutputCol("words")

    //用分词器对原始数据进行转换，生成新的一列words
    val wordsData = tokenizer.transform(movieDF)
//    wordsData.show()
    //引入HashingTF工具，可以把一个词语序列转换成对应的词频   设置hash分桶的数量为50，太小容易出现哈希碰撞
    val hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(50)
    val featurizeData = hashingTF.transform(wordsData)
//    featurizeData.show(truncate = false)
    //结果为下面所示  |表示一列的开始
    //|2  |Jumanji (1995)    |Adventure Children Fantasy    |[adventure, children, fantasy]   |(50,[11,13,19],[1.0,1.0,1.0])
    //引入IDF工具，可以得到idf模型
    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    //训练idf模型，得到每个词的逆文档频率
    val idfModel = idf.fit(featurizeData)
    //用模型对原数据进行处理，得到文档中每个词的tf-idf作为新的特征向量
    val rescaleData = idfModel.transform(featurizeData)
//    rescaleData.show(truncate = false)
    val movieFeatures = rescaleData.map(
      func = row => (row.getAs[Int]("mid"), row.getAs[SparseVector]("features").toArray)
    )
      .rdd
      .map(
        x => (x._1,new DoubleMatrix(x._2))
      )
 //  movieFeatures.show(false)      //在转换成RDD之前使用这一步进行打印
//    movieFeatures.collect().foreach(println)    //转换成RDD之后需要这样打印
//    //对所有电影两两计算它们的相似度，先做笛卡尔积
    val movieRecs = movieFeatures.cartesian(movieFeatures)
      .filter{
          //把自己跟自己的配对过滤掉
        case (a,b) => a._1 != b._1
      }
      .map{
        case (a,b)=>{
          val simScore = this.consinSim(a._2,b._2)
          (a._1,(b._1,simScore))
        }
      }
      .filter(_._2._2 > 0.6)//过滤出相似度大于0.6的
      .groupByKey()
      .map{
        case (mid,items) => MovieRecs(mid,items.toList.sortWith(_._2>_._2).map(x => Recommendation(x._1,x._2)))
      }
      .toDF()
    movieRecs.show(false)
//    movieRecs.write
//      .option("uri",mongoConfig.uri)
//      .option("collection",CONTENT_MOVIE_RECS)
//      .mode("overwrite")
//      .format("com.mongodb.spark.sql")
//      .save()
    spark.stop()

  }
  //求向量余弦相似度
  def consinSim(movie1: DoubleMatrix, movies2: DoubleMatrix):Double={
    movie1.dot(movies2)/(movie1.norm2()*movies2.norm2())
  }

}























