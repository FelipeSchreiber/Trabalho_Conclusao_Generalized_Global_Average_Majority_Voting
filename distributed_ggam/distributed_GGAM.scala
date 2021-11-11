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
	def writeToFile(id:Long,arr:Array[Double],file:PrintWriter)
	{
		file.write(String.valueOf(id))
		var tempStr = ""
		for(j <- 0 to arr.length-1) 
		{
          tempStr = "," + String.valueOf(arr(j))
          file.write(tempStr)
		}
		file.write("\n")
	}

	val V = graph.numVertices.toInt
    val vertexBelonging = new Array[Double](V)
	vertexBelonging.map{_ => 0}
	def mappingLabels(vid:VertexId): Array[Double] = 
	{
		var myBelonging = new Array[Double](V)
		myBelonging.map{_ => 0}
		myBelonging(vid.toInt) = 1
		return myBelonging
	}
    //LPA algorithm
    val lpaGraph: Graph[Array[Double],Int] = graph.mapVertices { 
		case (vid, _) =>  mappingLabels(vid)
	}
    
    type Label = VertexId
	val initialMessage = new Array[Double](V)
	initialMessage.map{_ => 0}
	var g = new Array[Double](V)
	g.map{_ => 1/V}

	def sendMsg(e: EdgeTriplet[Array[Double],Int]): Iterator[(VertexId,Array[Double])] =
		Iterator((e.srcId, e.dstAttr), 
		(e.dstId,e.srcAttr))
		
	def vprog(vid: VertexId, attr: Array[Double], message: Array[Double]): Array[Double] = 
	{
		var label = new Array[Double](V)
		label.map{_ => 0}
		var total = message.sum
		message.map{t => t/total}
		if (message.isEmpty) 
			return attr 
		else 
		{
			label = message.zip(g).map{
				case(x,y) => x*y
			}
		}
		return label
	}

	def mergeMsg(arr1: Array[Double], arr2: Array[Double]): Array[Double] = 
	{
		var aggArrays = arr1.zip(arr2).map{case(x,y) => x + y}
		return aggArrays
	}

	var i = 0
	while(i < maxIter)
	{
		lpaGraph.pregel(initialMessage, step)(vprog, sendMsg, mergeMsg)
		lpaGraph.cache() 
		i += step
		g.map{_ => 0}
		lpaGraph.vertices.take(3).foreach(a => g.zip(a._2).map{case(x,y) => x + y})
		var total = g.sum
		g.map{t => 1/total}
	}
	val writer = new PrintWriter(new File("results.txt"))
	lpaGraph.vertices.take(10).map{case(id,it) => writeToFile(id,it,writer)}
	writer.close()
  }
}
