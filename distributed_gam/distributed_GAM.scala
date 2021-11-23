import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
import org.apache.spark.graphx._
import org.apache.spark._
import java.io._

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
	def writeToFile(id:Long,label:Long,file:PrintWriter)
	{
		file.write(String.valueOf(id))
		var tempStr = ""
		tempStr = "," + String.valueOf(label)
        file.write(tempStr)
		file.write("\n")
	}
    
    //LPA algorithm
	//val r = new scala.util.Random
	//r.nextInt(1).toLong
    var lpaGraph = graph.mapVertices { case (vid, _) =>  vid%2}
    val V = graph.numVertices
    
    type Label = VertexId
	val initialMessage = Map[Label, Long]()
	val g = sc.doubleAccumulator("my accum")
	lpaGraph.vertices.foreach{a => g.add(a._2.toDouble/V)}
	println("Initial G value: "+g.value.toString)
	var g_broadcast = sc.broadcast(g.value)
	
	def sendMsg(e: EdgeTriplet[Label, Int]): Iterator[(VertexId,Map[Label, Long])] =
		Iterator((e.srcId, Map(e.dstAttr -> 1L)), 
		(e.dstId,Map(e.srcAttr -> 1L)))

	def vprog(vid: VertexId, attr: Long, message: Map[Label, Long]): Long = 
	{
		var label: Long = 0
		if (message.isEmpty) 
			return attr 
		else 
		{
			var localAvg:Double = message.getOrElse(1,0L).toDouble
			localAvg /= (message.getOrElse(0,0L) + message.getOrElse(1,0L)).toDouble
			//println("Local avg "+vid.toString+" "+localAvg.toString) 
			if (localAvg > g_broadcast.value)
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
		lpaGraph = lpaGraph.pregel(initialMessage, step)(vprog, sendMsg, mergeMsg)
		i += step
		g.reset
		println("G value: "+g.value.toString)
		lpaGraph.vertices.foreach(a => g.add(a._2.toDouble/V))
		println("G value: "+g.value.toString)
		g_broadcast = sc.broadcast(g.value)
		println(i)
	}
	val writer = new PrintWriter(new File("results.txt"))
	lpaGraph.vertices.collect.map{case(id,it) => writeToFile(id,it,writer)}
	writer.close()
  }
}
