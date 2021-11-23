import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
import org.apache.spark.graphx._
import org.apache.spark._
import org.apache.spark.util.AccumulatorV2
import java.io._

//Para compilar, faça sbt assembly. Para executar, $SPARKHOME/spark-submit --class "Nome_da_classe" caminho_ate_o_jar
object SimpleGraphApp {
  def main(args: Array[String])
  {
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
		//file.write(String.valueOf(id))
		var tempStr = ""
		tempStr = String.valueOf(arr(0))
        file.write(tempStr)
		for(j <- 1 to arr.length-1) 
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
    var lpaGraph: Graph[Array[Double],Int] = graph.mapVertices { 
		case (vid, _) =>  mappingLabels(vid)
	}
    
    type Label = VertexId
	val initialMessage = new Array[Double](V)
	initialMessage.map{_ => 0}
	var g_init = new Array[Double](V)
	g_init = g_init.map{_ => 1.0/(2*V)}

	class arrayAccumulator(var arrValue: Array[Double]) extends AccumulatorV2[Array[Double],Array[Double]] 
	{
		override def add(v: Array[Double]): Unit = v.zipWithIndex.foreach(t => arrValue(t._2) += t._1)
  		override def value: Array[Double] = arrValue
		override def isZero: Boolean = true
		override def copy(): AccumulatorV2[Array[Double],Array[Double]] = new arrayAccumulator(arrValue).asInstanceOf[AccumulatorV2[Array[Double],Array[Double]]]
		override def reset(): Unit = Array.empty[Double]
		override def merge(other: AccumulatorV2[Array[Double],Array[Double]]): Unit = arrValue ++ other.value
	}

	var g = new arrayAccumulator(g_init)
	sc.register(g, "array Accum")
	var g_broadcast = sc.broadcast(g.value)

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
			label = message.zip(g_broadcast.value.asInstanceOf[Array[Double]]).map
			{
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
	var k = 0
	var sum: Double = 0.0
	while (k < g.value.length) 
	{
		sum += g.value(k)
		k += 1
	}
	println("Valor inicial: "+sum)
	lpaGraph.vertices.collect.foreach(x => x._2.foreach(y => sum+=y))
	println("Valor inicial LPA: "+sum) 
	// while(i < maxIter)
	// {
	// 	lpaGraph = lpaGraph.pregel(initialMessage, step)(vprog, sendMsg, mergeMsg)
	// 	i += step
	// 	g.value.asInstanceOf[Array[Double]].map{_ => 0.0}
	// 	lpaGraph.vertices.foreach(a => g.value.asInstanceOf[Array[Double]].zip(a._2).map{case(x,y) => x + y})
	// 	k=0
	// 	sum=0.0
	// 	while (k < g.value.length) 
	// 	{			
	// 		sum += g.value(k)
	// 		k += 1
	// 	}
	// 	println(sum)
	// 	var total = g.value.asInstanceOf[Array[Double]].sum
	// 	g.value.asInstanceOf[Array[Double]].map{t => 1/total}
	// 	g_broadcast = sc.broadcast(g.value)
	// }
	// //val writer = new PrintWriter(new File("results.txt"))
	// //lpaGraph.vertices.collect.sortBy(_._1).map{case(id,it) => writeToFile(id,it,writer)}
	// //writer.close()
	// sum = 0.0
	// lpaGraph.vertices.collect.foreach(x => x._2.foreach(y => sum+=y)) 
	// println(sum)
  }
}
