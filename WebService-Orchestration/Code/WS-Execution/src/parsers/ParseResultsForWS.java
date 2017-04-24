package parsers;

import java.util.ArrayList;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.xpath.XPath;
import javax.xml.xpath.XPathConstants;
import javax.xml.xpath.XPathFactory;

import org.w3c.dom.Document;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

import download.WebService;

public class ParseResultsForWS  {
	 static final DocumentBuilderFactory builderFactory = DocumentBuilderFactory.newInstance();
     static final DocumentBuilder builder = getBuilder();
     static final XPath xPath =  XPathFactory.newInstance().newXPath();

     public static final  DocumentBuilder getBuilder(){
    	     try{
    	    	 	return builderFactory.newDocumentBuilder();
    	     }catch(Exception e ){ return null;}
     }
     
     /**
      * 
      * @param fileWithWithTransfResults
      * @param ws
      * @return the list of tuples; each tuples respects the order of head variables as defined in the description of the WS
      * @throws Exception
      */
	public static ArrayList<String[]> showResults(String fileWithWithTransfResults, WebService ws) throws Exception{
		ArrayList<String[]> listOfTupleResults= new ArrayList<String[]>();
		
		 Document xmlDocument = builder.parse(fileWithWithTransfResults);
		 System.out.println("Parse document "+fileWithWithTransfResults);
		 
		 String record = "/RESULT/RECORD";
         NodeList nodeList = (NodeList) xPath.compile(record).evaluate(xmlDocument, XPathConstants.NODESET);
         for (int i = 0; i < nodeList.getLength(); i++) {
        	 	// init the new tuple vector 
        	 	String [] tuple = new String[ws.headVariables.size()]; 
        	 	for(int k=0;k<tuple.length;k++){
        	 		tuple[k]=null;
        	 	}
        			 
        	 	
        	    //read each item (value of a variable)
        	 	String item_expr= "./ITEM";
        	 	
        	 	NodeList listItem = (NodeList) xPath.compile(item_expr).evaluate(nodeList.item(i), XPathConstants.NODESET);
            for (int j = 0; j < listItem.getLength(); j++) {
            		String value=listItem.item(j).getTextContent();
            	
            		String exprVarible="./@ANGIE-VAR";
            		String variable=((Node) xPath.compile(exprVarible).evaluate(listItem.item(j), XPathConstants.NODE)).getNodeValue();
            		
            		Integer posVariable=ws.headVariableToPosition.get(variable.trim());
            		if(posVariable==null) System.err.println("Incorrect script: variable unknown ");
            		tuple[posVariable]=value.trim();
            }
        	 	
        	 	listOfTupleResults.add(tuple);
         }
		 
		 return listOfTupleResults;
	}
	
}
