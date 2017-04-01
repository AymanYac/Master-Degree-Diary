import java.io.IOException;
import java.util.StringTokenizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.MapWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class CWBTIAB {
	
	public static class StripesMapper extends Mapper<LongWritable,Text,Text,MapWritable>{
		
		private Text word = new Text();
		private final static LongWritable one = new LongWritable(1);
		
		public void map(LongWritable key,Text value,Context context) throws IOException, InterruptedException {
			StringTokenizer itr = new StringTokenizer(value.toString(),",");
			StringTokenizer itr2 = new StringTokenizer(value.toString(),",");
			while(itr.hasMoreElements()){
				String itm = itr.nextElement().toString();
				MapWritable map = new MapWritable();
				while(itr2.hasMoreElements()){
					String itm2 = itr2.nextElement().toString();
					if(!itm2.equals(itm)){
						if(!map.containsKey(itm2)){
						word.set(itm2.replace(" ", ""));
						map.put(word,one);
						
						}else{
							LongWritable val = (LongWritable) map.get(itm2);
							int valeur = (int) val.get();
							valeur++;
							val = new LongWritable(valeur);
							word.set(itm2.replace(" ", ""));
							map.replace(word, (Writable) val);
							}
						}
					}
				word.set(itm.replace(" ", ""));
				context.write(word, map);
				
			}
		}
	}
		
		public static class DeserializerReducer extends Reducer<Text,MapWritable,Text,MapWritable>{
			public void reduce(Text itm,Iterable<MapWritable> stripes,Context context) throws IOException, InterruptedException{
				MapWritable out = new MapWritable();
				for (MapWritable map : stripes){
					for (Map.Entry<Writable,Writable> e : map.entrySet()){
						Writable key = e.getKey();
						if(out.containsKey(key)){
							LongWritable val = (LongWritable) e.getValue();
							int valeur = (int) val.get();
							val = (LongWritable) out.get(key);
							int valeur2 = (int) val.get();
							valeur = valeur + valeur2;
							val = new LongWritable(valeur);
							out.put(key, val);
							
						}
						else{
							out.put(key, e.getValue());
						}
					}
				}
			context.write(itm, out);
			}
		}

		
		
		public static void main(String[] args) throws Exception {
		    Configuration conf = new Configuration();
		    Job job = Job.getInstance(conf, "CWBTIAB");
		    job.setJarByClass(CWBTIAB.class);
		    job.setMapperClass(StripesMapper.class);
		    //job.setCombinerClass(DeserializerReducer.class);
		    job.setReducerClass(DeserializerReducer.class);
		    job.setMapOutputKeyClass(Text.class);
		    job.setMapOutputValueClass(MapWritable.class);
		    job.setOutputKeyClass(Text.class);
		    job.setOutputValueClass(MapWritable.class);
		    FileInputFormat.addInputPath(job, new Path(args[0]));
		    FileOutputFormat.setOutputPath(job, new Path(args[1]));
		    System.exit(job.waitForCompletion(true) ? 0 : 1);
		    //comment
		  }
}