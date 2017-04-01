import java.io.IOException;
import java.util.Set;

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
		private Text tmp = new Text();
		private LongWritable tmpn = new LongWritable();
		private final static LongWritable one = new LongWritable(1);
		MapWritable map = new MapWritable();
		String out;
		public void map(LongWritable key,Text value,Context context) throws IOException, InterruptedException {
			String [] tokens = value.toString().replaceAll("\\s+","").split(",");
			String [] tokens2 = value.toString().replaceAll("\\s+","").split(",");
			for (String tkn : tokens){
				map.clear();
				for (String tkn2 : tokens2){
					if(tkn2 != "" && tkn2 != tkn){
						tmp.set(tkn2);
						if (map.containsKey(tmp)){
							tmpn=(LongWritable) map.get(tmp);
							Long tn = tmpn.get();
							tn++;
							tmpn= new LongWritable(tn);
							map.replace(tmp, tmpn);
						}
						else{
							map.put(tmp,one);
						}
					}
				}
				word.set(tkn);
				context.write(tmp, map);
			}
			
			
		}
	}
		
		public static class DeserializerReducer extends Reducer<Text,MapWritable,Text,Text>{
			private MapWritable incrementingMap = new MapWritable();
			private Text arg1 = new Text();
			private long temp = new Integer(0);
			private String str = new String();
			private LongWritable one = new LongWritable(1);
			private LongWritable two = new LongWritable(2);

			public void reduce(Text itm,Iterable<MapWritable> stripes,Context context) throws IOException, InterruptedException{
				/*incrementingMap.clear();
				for (MapWritable map : stripes){
					Set <Writable> keys = map.keySet();
					for (Writable key : keys){
						if(incrementingMap.containsKey(key)){
							LongWritable tmp = (LongWritable) incrementingMap.get(key);
							temp = tmp.get();
							temp++;
							tmp = new LongWritable(temp);
							incrementingMap.replace(key, tmp);
						}
						else{
							incrementingMap.put(key, one);
						}
					}
				}
				Set <Writable> keys = incrementingMap.keySet();
				for (Writable key : keys){
					str += ((Text) key).toString();
					str += " > ";
					LongWritable tmp = (LongWritable) incrementingMap.get(key);
					temp = tmp.get();
					str += Long.toString(temp);
					str += ",";
				}
				arg1.set(str);*/
				
				////////DEBUGING
				str="";
				for (MapWritable map : stripes){
					Set<Writable> keys = map.keySet();
					for (Writable key : keys){
						str+=((Text) key).toString()+" > ";
						str+=Long.toString(((LongWritable) map.get(key)).get());
						str+=",";
					}
				}
				arg1.set(str);
				context.write(itm, arg1);
				
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
		    job.setOutputValueClass(Text.class);
		    FileInputFormat.addInputPath(job, new Path(args[0]));
		    FileOutputFormat.setOutputPath(job, new Path(args[1]));
		    System.exit(job.waitForCompletion(true) ? 0 : 1);
		    //comment
		  }
}