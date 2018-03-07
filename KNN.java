import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.nio.file.FileStore;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Set;

import org.apache.hadoop.io.LongWritable; 
import org.apache.hadoop.conf.Configuration;  
import org.apache.hadoop.fs.Path;  
import org.apache.hadoop.io.IntWritable; 
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FSDataInputStream;
import java.net.URI;

//本程序的目的是实现MR-K-NN算法
public class KNN{   
      // 传入main的arg[]为在
      // linux的CLI下输入的参数
      //arg[0]为输入路径
      //arg[1]为输出路径
      //训练数据已在hdfs中
      public static void main(String[] args) throws Exception{
            //获取文件对象
          FileSystem fileSystem = FileSystem.get(new Configuration());
            //检查输出目录是否存在
          if(fileSystem.exists(new Path(args[1]))){
            //存在则删除目录和内容
            //清空的作用
              fileSystem.delete(new Path(args[1]), true);
          }

          Job job = new Job(new Configuration(),"KNN");
          job.setJarByClass(KNN.class);
          job.setMapperClass(MyMapper.class);
          job.setReducerClass(MyReducer.class);
          
          job.setOutputKeyClass(Text.class);
          job.setOutputValueClass(Text.class);
          FileOutputFormat.setOutputPath(job, new Path(args[1]));
          //在这里指定输入文件的父目录即可，MapReduce会自动读取输入目录下所有的文件
          FileInputFormat.setInputPaths(job, new Path(args[0]));
          System.exit(job.waitForCompletion(true)?0:1);
      }

      //Map过程
     public static class MyMapper extends Mapper<Object, Text, Text, Text>{
           //k近邻
           public int k = 5;//k在这里可以根据KNN算法实际要求取值
           private ArrayList<tuple>trainData;
           private ArrayList<Double>distance;
           private ArrayList<String>correspondingFlag;
           protected void map(Object k1, Text v1,Context context)throws IOException, InterruptedException{
                 init();//分配ArrayList空间
                 readTrain();//读取训练集数据

                 //读取测试信息
                 tuple test=new tuple();
                 test.read(v1.toString());

                 //插入排序压入测试数据

                 //一个尾数据
                 distance.add(Double.MAX_VALUE);
                 correspondingFlag.add("-1");
                 
                 for(tuple thisTuple:trainData){
                       Double thisDistance=getDistance(thisTuple.getData(), test.getData());
                       //插入排序
                       int pt=0;
                       while(thisDistance>distance.get(pt))pt++;
                       distance.add(pt, thisDistance);
                       correspondingFlag.add(pt, thisTuple.getFlag());
                 }

                 //删除k之后的元素
                 while(distance.size()>k){
                       distance.remove(k);
                       correspondingFlag.remove(k);
                 }

                 //写出结果
                 for(String thisFlag:correspondingFlag){
                       context.write(new Text(v1.toString()),new Text(thisFlag));
                 }
          }
          private void init(){
                trainData=new ArrayList<>();
                distance=new ArrayList<>();
                correspondingFlag=new ArrayList<>();
          }
          //读取训练数据
          private void readTrain()throws IOException, InterruptedException{
                FileSystem fileSystem = null;
                try{
                      fileSystem = FileSystem.get(new URI("hdfs://AroyaMaster:9000/"), new Configuration());
                }
                catch(Exception e){}
                //建立读取关系
                FSDataInputStream fr0 = fileSystem.open(new Path("hdfs://AroyaMaster:9000/train.txt"));
                //读取数据
                BufferedReader fr1 = new BufferedReader(new InputStreamReader(fr0));
                
                //读取整行
                String str = fr1.readLine();
                while(str!=null){
                      //创建元组，将整行数据读入元组
                      tuple thisTuple=new tuple();
                      thisTuple.read(str);
                      trainData.add(thisTuple);
                      //读取下一行
                      str = fr1.readLine();
                }
          }
            //计算两个数组的距离
            private double getDistance(ArrayList<Double>a,ArrayList<Double>b){
                  //最小size
                  int size=(a.size()>b.size()?b.size():a.size());
                  double ans=0;
                  //求距离
                  for(int i=0;i<size;i++){
                        ans+=Math.pow(a.get(i)-b.get(i),2);
                  }
                  return Math.sqrt(ans);
            }
     }

     public static class MyReducer extends Reducer<Text, Text, Text, Text>{

            String ans;  
            ArrayList<String> flags;

           //k2为测试数据的Text
           //v2s为测试数据对应的Text的数组（Flag）
           //对应mapper中 context.write([k2],[v2s[i]]);
            protected void reduce(Text k2, Iterable<Text> v2s,Context context)throws IOException, InterruptedException{
                  flags=new ArrayList<>();

                  //统计出现次数
                  HashMap<String,Integer>Counter=new HashMap<>();
                  for(Text thisText:v2s){
                        if(Counter.containsKey(thisText.toString())){
                              Counter.replace(thisText.toString(), 
                                    Counter.get(thisText.toString())+1);
                        }
                        else Counter.put(thisText.toString(), 1);
                  }

                  //找出次数最大的
                  Set<String> keySet=Counter.keySet();
                  String ans="";
                  Integer nowTimes=Integer.MIN_VALUE;
                  for(String thisFlag:keySet){
                        if(Counter.get(thisFlag)>nowTimes){
                              ans=thisFlag;
                              nowTimes=Counter.get(thisFlag);
                        }
                  }
                  //写出数据k2+Flag
                  context.write(new Text(k2),new Text(ans));
            } 
     }
}
class tuple{
      ArrayList<Double> data;
      String Flag;
      public tuple(){
            data=new ArrayList<>();
            //nothing
      }
      public void read(String str){
            //分割数据
            String[] split=str.split("\t");
            int last=split.length-1;
            //存放数据
            for(int i=0;i<last;i++){
                  data.add(Double.parseDouble(split[i]));
            }
            //最后一位是flag
            Flag=split[last];
      }
      public ArrayList<Double>getData(){return data;}
      public String getFlag(){return Flag;}
}