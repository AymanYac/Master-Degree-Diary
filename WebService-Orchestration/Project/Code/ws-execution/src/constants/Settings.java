package constants;

import java.io.File;

public class Settings {

	public static final String rootProject="/home/moriarty/Desktop/DK-II/Code/Code/WS-Evaluation/";
	public static final String dirWithDef=rootProject+"ws-definitions/";
	
	
	public static final String getDirForCallResults(String ws){
		File f = new File(rootProject+ws+"/call_results/");
		if(f.isDirectory()){
			return rootProject+ws+"/call_results/";
		}
		//create folder for newly called WS
		f.mkdirs();
		return rootProject+ws+"/call_results/";
	}
	
	public static final String getDirForTransformationResults(String ws){
		File f = new File(rootProject+ws+"/transf_results/");
		if(f.isDirectory()){
			return rootProject+ws+"/transf_results/";
		}
		//create folder for newly called WS
		f.mkdirs();
		return rootProject+ws+"/transf_results/";
	}
	
	
	
}
