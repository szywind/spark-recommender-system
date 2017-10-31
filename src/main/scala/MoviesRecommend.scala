import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by a on 25/07/2017.
  */
object MoviesRecommend {
  def main(args: Array[String]) {
    if (args.length < 2) {
      System.err.println("Usage : <master> <hdfs dir path>")
      System.exit(1)
    }

    //屏蔽日志，由于结果是打印在控制台上的，为了方便查看结果，将spark日志输出关掉
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)

    //创建入口对象
    val conf = new SparkConf().setMaster(args(0)).setAppName("Collaborative Filtering")
    val sc = new SparkContext(conf)

    //评分训练总数据集，元组格式
    val ratingsList_Tuple = sc.textFile(args(1) + "/ratings.dat").map { lines =>
      val fields = lines.split("::")
      (fields(0).toInt, fields(1).toInt, fields(2).toDouble, fields(3).toLong % 10)//这里将timespan这列对10做取余操作，这样一来个评分数据的这一列都是一个0-9的数字，做什么用？接着看下面
    }

    //评分训练总数据集，模拟键值对形式，键是0-9中的一个数字，值是Rating类型
    val ratingsTrain_KV = ratingsList_Tuple.map(x =>
      (x._4, Rating(x._1, x._2, x._3)))
    //打印出从ratings.dat中，我们从多少个用户和电影之中得到了多少条评分记录
    println("get " + ratingsTrain_KV.count()
      + " ratings from " + ratingsTrain_KV.map(_._2.user).distinct().count()
      + "users on " + ratingsTrain_KV.map(_._2.product).distinct().count() + "movies")

    //我的评分数据，RDD[Rating]格式
    val myRatedData_Rating = sc.textFile(args(1) + "/test.dat").map { lines =>
      val fields = lines.split("::")
      Rating(fields(0).toInt, fields(1).toInt, fields(2).toDouble)
    }

    //从训练总数据总分出80%作为训练集，20%作为验证数据集，20%作为测试数据集，前面的将timespan对10做取余操作的作用就是为了从总数据集中分出三部分
    //设置分区数
    val numPartitions = 3
    //将键的数值小于8的作为训练数据
    val traningData_Rating = ratingsTrain_KV.filter(_._1 < 6)
      .values//注意，由于原本的数据集是伪键值对形式的，而当做训练数据只需要RDD[Rating]类型的数据，即values集合
      .union(myRatedData_Rating)//使用union操作将我的评分数据加入训练集中，以做为训练的基准
      .repartition(numPartitions)
      .cache()

    //格式和意义和上面的类似，由于是验证数据，并不需要我的评分数据，所以不用union
    val validateData_Rating = ratingsTrain_KV.filter(x => x._1 >= 6 && x._1 < 8)
      .values
      .repartition(numPartitions)
      .cache()
    val testData_Rating = ratingsTrain_KV.filter(_._1 >= 8)
      .values
      .repartition(numPartitions)
      .cache()

    //打印出用于训练，验证和测试的数据集分别是多少条记录
    println("training data's num : " + traningData_Rating.count()
      + " validate data's num : " + validateData_Rating.count()
      + " test data's num : " + testData_Rating.count())

    //开始模型训练，根据方差选择最佳模型
    val ranks = List(8, 22)
    val lambdas = List(0.1, 10.0)
    val iters = List(5, 7)//这里的迭代次数要根据各自集群机器的硬件来选择，由于我的机器不行最多只能迭代7次，再多就会内存溢出
    var bestModel: MatrixFactorizationModel = null
    var bestValidateRnse = Double.MaxValue
    var bestRank = 0
    var bestLambda = -1.0
    var bestIter = -1
    //一个三层嵌套循环，会产生8个ranks ，lambdas ，iters 的组合，每个组合都会产生一个模型，计算8个模型的方差，最小的那个记为最佳模型
    for (rank <- ranks; lam <- lambdas; iter <- iters) {
      val model = ALS.train(traningData_Rating, rank, iter, lam)
      //rnse为计算方差的函数，定义在最下方
      val validateRnse = rnse(model, validateData_Rating, validateData_Rating.count())
      println("validation = " + validateRnse
        + " for the model trained with rank = " + rank
        + " lambda = " + lam
        + " and numIter" + iter)
      if (validateRnse < bestValidateRnse) {
        bestModel = model
        bestValidateRnse = validateRnse
        bestRank = rank
        bestLambda = lam
        bestIter = iter
      }
    }

    //将最佳模型运用在测试数据集上
    val testDataRnse = rnse(bestModel, testData_Rating, testData_Rating.count())
    println("the best model was trained with rank = " + bestRank + " and lambda = " + bestLambda
      + " and numIter = " + bestIter + " and Rnse on the test data is " + testDataRnse)

    //计算和原先基础的相比其提升了多少
    val meanRating = traningData_Rating.union(validateData_Rating).map(_.rating).mean()
    val baseLineRnse = math.sqrt(testData_Rating.map(x => (meanRating - x.rating) * (meanRating - x.rating)).mean())
    val improvent = (baseLineRnse - testDataRnse) / baseLineRnse * 100
    println("the best model improves the baseline by " + "%2.2f".format(improvent) + "%")

    //电影列表总数据，元组格式
    val movieList_Tuple = sc.textFile(args(1) + "/movies.dat").map { lines =>
      val fields = lines.split("::")
      (fields(0).toInt, fields(1), fields(2))
    }

    //电影名称总数据，Map类型，键为id，值为name
    val movies_Map = movieList_Tuple.map(x =>
      (x._1, x._2)).collect().toMap

    //电影类型总数据，Map类型，键为id，值为type
    val moviesType_Map = movieList_Tuple.map(x =>
      (x._1, x._3)).collect().toMap

    var i = 1
    println("movies recommond for you:")
    //得到我已经看过的电影的id
    val myRatedMovieIds = myRatedData_Rating.map(_.product).collect().toSet
    //从电影列表中将这些电影过滤掉，剩下的电影列表将被送到模型中预测每部电影我可能做出的评分
    val recommondList = sc.parallelize(movies_Map.keys.filter(myRatedMovieIds.contains(_)).toSeq)
    //将结果数据按评分从大小小排序，选出评分最高的10条记录输出
    bestModel.predict(recommondList.map((0, _))).collect().sortBy(-_.rating).take(10).foreach { r =>
      println("%2d".format(i) + "----------> : \nmovie name --> "
        + movies_Map(r.product) + " \nmovie type --> "
        + moviesType_Map(r.product))
      i += 1
    }

    //计算可能感兴趣的人
    println("you may be interested in these people : ")
    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._
    //将电影，用户，评分数据转换成为DataFrame，进行SparkSQL操作
    val movies = movieList_Tuple
      .map(m => Movies(m._1.toInt, m._2, m._3))
      .toDF()

    val ratings = ratingsList_Tuple
      .map(r => Ratings(r._1.toInt, r._2.toInt, r._3.toInt))
      .toDF()

    val users = sc.textFile(args(1) + "/users.dat").map { lines =>
      val fields = lines.split("::")
      Users(fields(0).toInt, fields(2).toInt, fields(3).toInt)
    }.toDF()

    ratings.filter('rating >= 5)//过滤出评分列表中评分为5的记录
      .join(movies, ratings("movieId") === movies("id"))//和电影DataFrame进行join操作
      .filter(movies("mType") === "Drama")//筛选出评分为5，且电影类型为Drama的记录（本来应该根据我的评分数据中电影的类型来进行筛选操作，由于数据格式的限制，这里草草的以一个Drama作为代表）
      .join(users, ratings("userId") === users("id"))//对用户DataFrame进行join
      .filter(users("age") === 18)//筛选出年龄=18（和我的信息一致）的记录
      .filter(users("occupation") === 15)//筛选出工作类型=18（和我的信息一致）的记录
      .select(users("id"))//只保存用户id，得到的结果为和我的个人信息差不多的，而且喜欢看的电影类型也和我差不多 的用户集合
      .take(10)
      .foreach(println)
  }

  //计算方差函数
  def rnse(model: MatrixFactorizationModel, predictionData: RDD[Rating], n: Long): Double = {
    //根据参数model，来对验证数据集进行预测
    val prediction = model.predict(predictionData.map(x => (x.user, x.product)))
    //将预测结果和验证数据集join之后计算评分的方差并返回
//    val t = predictionData.map{case Rating(user,product,rating)=>((user,product),rating)}
    val predictionAndOldRatings = prediction.map(x => ((x.user, x.product), x.rating))
//          .join(t).values
      .join(predictionData.map(x => ((x.user, x.product), x.rating))).values
    math.sqrt(predictionAndOldRatings.map(x => (x._1 - x._2) * (x._1 - x._2)).reduce(_ - _) / n)
  }

  //样例类，用作SparkSQL隐式转换
  case class Ratings(userId: Int, movieId: Int, rating: Int)

  case class Movies(id: Int, name: String, mType: String)

  case class Users(id: Int, age: Int, occupation: Int)

}
