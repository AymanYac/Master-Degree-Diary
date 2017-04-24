package download;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.StringWriter;
import java.io.Writer;
import java.net.URL;
import java.net.URLConnection;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import javax.xml.transform.Result;
import javax.xml.transform.Source;
import javax.xml.transform.Transformer;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.stream.StreamResult;
import javax.xml.transform.stream.StreamSource;

//import apple.laf.JRSUIConstants.SegmentTrailingSeparator;
import constants.Formating;
import constants.Settings;


public class WebService {

	
	public String name;
	
	/** the lists of fragments and variables that form the URL, on the position of variables we have the value null  **/
	public List<String> urlFragments;
	
	/** The head **/
	public ArrayList<String> headVariables;  /** the order of the variables matters! 
								The first variables should be the input variables 
								and their order should be exactly the one required by the construction of the URL 
								**/
	public HashMap<String, Integer> headVariableToPosition;
	public int numberOfInputs;
	
	/** The body **/
	HashMap<String,String> prefixes;
	List<Triple> body;

	public HashMap<String, Boolean> headVariableToType;
	
	public String mergeTags;

	/** the body currently not handled!! **/
	public WebService(String name, List<String> urlFragments, HashMap<String, String> prefixes, ArrayList<String> headVariables, HashMap<String, Integer> headVariableToPosition, int numberInputs,HashMap<String, Boolean> headVariableToType,String mergeTags){
		this.name=name;
		this.urlFragments=urlFragments;
		
		this.headVariables=headVariables;
		this.headVariableToPosition=headVariableToPosition;
		this.headVariableToType=headVariableToType;
		this.numberOfInputs=numberInputs;
		
		this.prefixes=prefixes;
		this.mergeTags = mergeTags;
	}
	
	/** constructor for convenience purposes -- if we do not want to read a description from a file **/
	public WebService(String name, List<String> params){
		this.name=name;
		this.urlFragments=params;
	}
	
	
	
	
	
	/**
	 * @param inputs
	 * @return the file where the call result is stored; returns null if some error occurs 
	 */
	public String getCallResult(boolean overwrite,boolean seek_join,String... inputs){
		String JoinfileWithCallResult=Settings.getDirForCallResults(this.name)+"JOIN.xml";
		String fileWithCallResult=Settings.getDirForCallResults(this.name)+Formating.getFileNameForInputs(inputs);
		
		//search first the result in the joined cache
		File f= new File(JoinfileWithCallResult);
		if(f.exists() && !overwrite && seek_join) {
			System.out.println("Partial join available: "+JoinfileWithCallResult);
			return JoinfileWithCallResult;
		}
		
		//search alternatively the result in the cache 
		f= new File(fileWithCallResult);
		if(f.exists() && !overwrite) {
			String URL=getURLForCallWithInputs(inputs);
			System.out.println("Cache available for "+URL);
			return fileWithCallResult;
		}
		
		//otherwise call the web service
		String URL=getURLForCallWithInputs(inputs);
		System.out.println("Requesting URL > "+URL);
		System.out.println(fileWithCallResult);
		String confirmationFileWithResult=downloadCallResults(URL, fileWithCallResult);	
		return confirmationFileWithResult;

	}
	
	/** 
	 * @param fileWithCallResult
	 * @return the path of the file where the result is stored
	 * @throws Exception
	 */
	public String getTransformationResult(String  fileWithCallResult) throws Exception{
		Source callResult = new StreamSource(new File(fileWithCallResult));
		
		Source xsl = new StreamSource(new File(Settings.dirWithDef+this.name+".xsl"));
		
		String fileName=fileWithCallResult.substring(fileWithCallResult.lastIndexOf("/")+1);
		String fileWithTransformationResult=Settings.getDirForTransformationResults(this.name)+fileName;
		System.out.println("File with the transformation result: "+fileWithTransformationResult);		
		Result trasformResult = new StreamResult(new File(fileWithTransformationResult)); 
		
		Transformer transformer = TransformerFactory.newInstance().newTransformer(xsl);
		transformer.transform(callResult , trasformResult); 
		 
		return fileWithTransformationResult;
	}
	
	
	/**
	 * @param inputs
	 * @return the URL of the call for the given inputs 
	 */
	public String getURLForCallWithInputs(String... inputs){
		int i=0;
		StringBuffer call=new StringBuffer();
		for(String p:urlFragments){
			if(p==null){
				if(i>=inputs.length) return null; //something wrong; insufficient number of input values
				call.append(Formating.transformStringForURL(inputs[i]));
				i++;
				//i must be incremented for following inputs to be inserted in following variables
			}
			else call.append(p);
		}
		
		return call.toString();
	}
	
	/**
	 * 
	 * @param URL
	 * @param fileForTheResults
	 * @return downloads the file and stores it in the file. If the call result is JSON, then transforms it to XML 
	 */
	public String downloadCallResults(String URL, String fileForTheResults) {
		String newLine = System.getProperty("line.separator");
		BufferedReader in=null;
        Writer writer = null;
        try
        {  
        		URL url = new URL(URL);
        		URLConnection conn = url.openConnection();
        		// fake a request coming from a browser in order to avoid error 403 (from discogs for instance)
        		conn.setRequestProperty("User-Agent", "Mozilla/5.0 (Windows; U; Windows NT 6.1; en-GB;     rv:1.9.2.13) Gecko/20101203 Firefox/3.6.13 (.NET CLR 3.5.30729)");
        		//conn.setRequestProperty("Accept-Charset", "UTF-8");
        		in = new BufferedReader(new InputStreamReader(conn.getInputStream()));
        		
        		// I remove empty lines until the v1 character 
            String line;
            while ((line = in.readLine()) != null ) {
                    line=line.trim();
                    if(!line.equals("")) break;
                }
            
            // create a string writer if JSON detected 
            boolean isJSONData=false;
            if(line!=null){
          	  	if(line.startsWith("{")) {
          	  		System.out.println("JSON detected");
          	  		writer = new StringWriter();
          	  		isJSONData=true;
          	  	}else {
          	  		writer = new FileWriter(fileForTheResults);
          	  	}
          	  	writer.write(line+newLine);
            }
            
            
            // write the rest of the input file 
            while ((line = in.readLine()) != null) {
          	  	   writer.write(line+newLine);
          	  	   System.out.println(line);
            }
            writer.flush();
            
            // if it's json data do the transformation 
            if(isJSONData) JSONToXML.transformToXML(((StringWriter)writer).toString(), fileForTheResults);    
        }catch(IOException e){ 
        					System.out.println("Error in the download "+URL);
        					return null;
        				}
        catch(Exception e ){
        		System.out.println("Error transformation "+URL);
			return null;	
        }
        finally
        {
            try
            {
                if(writer!=null) writer.close();
                if(in!=null) in.close();
            }
            catch (IOException e)
            {
                e.printStackTrace();
                return null;
            }
        }
        return fileForTheResults;
	}
	
}
