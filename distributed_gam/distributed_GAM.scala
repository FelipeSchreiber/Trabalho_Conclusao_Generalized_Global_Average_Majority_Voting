import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
import org.apache.spark.graphx._
import org.apache.spark._

//Para compilar, faça sbt assembly. Para executar, $SPARKHOME/spark-submit --class "Nome_da_classe" caminho_ate_o_jar
object SimpleGraphApp {
  def main(args: Array[String]){
    // Configure the program 
    val conf = new SparkConf()
          .setAppName("Tiny Social")
          .setMaster("local")
          .set("spark.driver.memory", "2G")
    val sc = new SparkContext(conf)
    val maxIter = args(0).toInt
    val step = args(1).toInt
    
    val filePath = "/home/felipe/UFRJ/TCC/spark-3.1.2-bin-hadoop3.2/data/graphx/"
    // Load some data into RDDs
    val graph = GraphLoader.edgeListFile(sc,filePath+"/ppg_net.txt")
    
    //LPA algorithm
    val lpaGraph = graph.mapVertices { case (vid, _) => vid%2 }
    val V = graph.numVertices
    
    type Label = VertexId
	val initialMessage = Map[Label, Long]()
	var g:Double = 0.5
	
	def sendMsg(e: EdgeTriplet[Label, Int]): Iterator[(VertexId,Map[Label, Long])] =
		Iterator((e.srcId, Map(e.dstAttr -> 1L)), 
		(e.dstId,Map(e.srcAttr -> 1L)))
		
	def vprog(vid: VertexId, attr: Long, message: Map[Label, Long]): VertexId = 
	{
		//var localMax = message.maxBy(_._2)._1
		var label = 0
		if (message.isEmpty) 
			return attr 
		else 
		{
			var localAvg:Long = 0
			message.foreach(x => localAvg += x._2.toLong)
			if (localAvg > g*V)
				label = 1
			else
				label = 0 
		}
		return label
	}

	def mergeMsg(count1: Map[Label, Long], count2: Map[Label, Long]): Map[VertexId, Long] = 
	{
		(count1.keySet ++ count2.keySet).map 
		{ i =>
			val count1Val = count1.getOrElse(i, 0L)
			val count2Val = count2.getOrElse(i, 0L)
			i -> (count1Val + count2Val)
		}.toMap
	}
	var i = 0
	while(i < maxIter)
	{
		lpaGraph.pregel(initialMessage, step)(vprog, sendMsg, mergeMsg)
		i += step
		g=0
		lpaGraph.vertices.foreach(a => g += a._2)
		g /= V
	}
	lpaGraph.vertices.take(10).map(println)
  }
}
