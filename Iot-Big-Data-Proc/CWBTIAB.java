import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.io.MapWritable;

public class CWBTIAB {
	public static class StripesMapper extends Mapper <LongWritable, Text, Text, MapWritable>{
		Text word = new Text();    
		public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
		    	String[] itr = value.toString().split(",");
		    	for (String item:itr){
		    		MapWritable  map = new MapWritable();
		    		for (String it:itr){
		    			map.put(it, map.get(it));
		    		}
		    		word.set(item);
		    		context.write(word,map);
		    	}
	}
}
	public static class DeserializerReducer extends Reducer<Text,Map<String,Integer>,String,String>{
		public void reduce(String key,Iterable<Map> values,Context context) throws IOException, InterruptedException {
			Map<String,Integer> out = new HashMap<String,Integer>();
			for (Map<String,Integer> map : values){
				for (Map.Entry<String, Integer> entry : map.entrySet()){
					out.put(entry.getKey(), out.get(entry.getKey())+entry.getValue());
				}
			}
			context.write(key,out.get(key).toString());
		}
	}
	public static void main(String[] args) throws Exception {
	    Configuration conf = new Configuration();
	    Job job = Job.getInstance(conf, "CWBTIAB");
	    job.setJarByClass(CWBTIAB.class);
	    job.setMapperClass(StripesMapper.class);
	    job.setCombinerClass(DeserializerReducer.class);
	    job.setReducerClass(DeserializerReducer.class);
	    job.setOutputKeyClass(Text.class);
	    job.setOutputValueClass(String.class);
	    FileInputFormat.addInputPath(job, new Path(args[0]));
	    FileOutputFormat.setOutputPath(job, new Path(args[1]));
	    System.exit(job.waitForCompletion(true) ? 0 : 1);
	  }
	}