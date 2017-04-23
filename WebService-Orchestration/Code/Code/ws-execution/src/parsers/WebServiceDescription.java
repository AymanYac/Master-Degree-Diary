package parsers;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import javax.smartcardio.ATR;
import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import javax.xml.xpath.XPath;
import javax.xml.xpath.XPathConstants;
import javax.xml.xpath.XPathExpressionException;
import javax.xml.xpath.XPathFactory;

import org.w3c.dom.Document;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;

import constants.Settings;
import download.WebService;
 
public class WebServiceDescription {
	
    public static WebService loadDescription(String webServiceName) {
    		/** prefixes **/
    		HashMap<String, String> prefixes=new HashMap<String, String>();
    		
    		/** head variables **/
    		HashMap<String, Integer> headVariableToPosition=new HashMap<String, Integer>();
    		ArrayList<String> headVariables= new ArrayList<String>();
    		int numberInputs=0;
    		List<String> urlFragments= new ArrayList<String>();
    	
        try {
            FileInputStream file = new FileInputStream(new File(Settings.dirWithDef+webServiceName+".xml"));
                 
            DocumentBuilderFactory builderFactory = DocumentBuilderFactory.newInstance();
             
            DocumentBuilder builder =  builderFactory.newDocumentBuilder();
             
            Document xmlDocument = builder.parse(file);
 
            XPath xPath =  XPathFactory.newInstance().newXPath();
 
            //parse prefixes
            String prefix = "/ws/prefix";
            NodeList nodeList = (NodeList) xPath.compile(prefix).evaluate(xmlDocument, XPathConstants.NODESET);
            for (int i = 0; i < nodeList.getLength(); i++) {
            		String expr_name = "./@name";
            		Node nodeName = (Node) xPath.compile(expr_name).evaluate(nodeList.item(i), XPathConstants.NODE);
            		String prefix_name=nodeName.getNodeValue();
            		
            		String expr_value = "./@value";
            		Node nodeValue = (Node) xPath.compile(expr_value).evaluate(nodeList.item(i), XPathConstants.NODE);
            		String prefix_value=nodeValue.getNodeValue();
            		prefixes.put(prefix_name.trim(), prefix_value.trim());
            		System.out.println(" prefix name="+prefix_name.trim()+"  value="+prefix_value.trim());
            		
            }
            
            //parse variables in the head: order matters!
            String headVariableExpr = "/ws/headVariables/variable";
            nodeList = (NodeList) xPath.compile(headVariableExpr).evaluate(xmlDocument, XPathConstants.NODESET);
            for (int i = 0; i < nodeList.getLength(); i++) {
            		String expr_name = "./@name";
            		String name=((Node) xPath.compile(expr_name).evaluate(nodeList.item(i), XPathConstants.NODE)).getNodeValue();
            		
            		String expr_type = "./@type";
            		String type=((Node) xPath.compile(expr_type).evaluate(nodeList.item(i), XPathConstants.NODE)).getNodeValue();
            		if(type.trim().startsWith("in")) numberInputs=i;
            		
            		headVariables.add(name.trim());
            		headVariableToPosition.put(name.trim(), new Integer(i));
            		
            		System.out.println("Variable : "+name+" position "+i);
            }
            
            
            //parse url fragments
            String exprURLFragments = "/ws/call/part";
            nodeList = (NodeList) xPath.compile(exprURLFragments).evaluate(xmlDocument, XPathConstants.NODESET);
            for (int i = 0; i < nodeList.getLength(); i++) {
            		
            		String expr_type = "./@type";
            		String type=((Node) xPath.compile(expr_type).evaluate(nodeList.item(i), XPathConstants.NODE)).getNodeValue();
            		if(type.trim().startsWith("input")) urlFragments.add(null);
            		else {
            			String expr_value = "./@value";
            			String fixPart=((Node) xPath.compile(expr_value).evaluate(nodeList.item(i), XPathConstants.NODE)).getNodeValue();
                		urlFragments.add(fixPart.trim());
            		}
            }
            System.out.print("The parts of the URLs (calls):");
            for(String part:urlFragments){
            		System.out.print(" "+part);
            }
            System.out.println("");
            
            return new WebService(webServiceName, urlFragments, prefixes, headVariables, headVariableToPosition, numberInputs);
            
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            return null;
        } catch (SAXException e) {
            e.printStackTrace();
            return null;
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        } catch (ParserConfigurationException e) {
            e.printStackTrace();
            return null;
        } catch (XPathExpressionException e) {
            e.printStackTrace();
            return null;
        } 
        
        
       
    }
}