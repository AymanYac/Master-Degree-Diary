import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;

import javax.xml.stream.XMLEventFactory;
import javax.xml.stream.XMLEventReader;
import javax.xml.stream.XMLEventWriter;
import javax.xml.stream.XMLInputFactory;
import javax.xml.stream.XMLOutputFactory;
import javax.xml.stream.XMLStreamException;
import javax.xml.stream.events.StartDocument;
import javax.xml.stream.events.XMLEvent;


public class XmlMerger {

	public String merge(HashSet<String> targetFiles, String... headers ) throws FileNotFoundException, XMLStreamException {
		 	XMLEventWriter eventWriter;
		    XMLEventFactory eventFactory;
		    XMLOutputFactory outputFactory = XMLOutputFactory.newInstance();
		    XMLInputFactory inputFactory = XMLInputFactory.newInstance();
		    String MergeFilePath = new File(targetFiles.iterator().next()).getParentFile().getAbsolutePath()+"/JOIN.xml";
		    eventWriter = outputFactory.createXMLEventWriter(new FileOutputStream(MergeFilePath));
		    eventFactory = XMLEventFactory.newInstance();
		    XMLEvent newLine = eventFactory.createDTD("\n");                
		    // Create and write Start Tag
		    StartDocument startDocument = eventFactory.createStartDocument();
		    eventWriter.add(startDocument);
		    eventWriter.add(newLine);
		    
		    //System.out.println(xmlfiles);
		    //String[] filenames = new String[]{"test1.xml", "test2.xml","test3.xml"};
		    Boolean first = true;
		    for(String filename:targetFiles){
		    	if(filename.equals(MergeFilePath)) {continue;} //we don't want the target file to be merged with itself
		    	
		    		XMLEventReader test = inputFactory.createXMLEventReader(filename,
		                             new FileInputStream(filename));
		        while(test.hasNext()){
		            XMLEvent event= test.nextEvent();
		            if(event.getEventType() == XMLEvent.START_ELEMENT){
		            	if(!first && Arrays.asList(headers).contains(event.toString().split("::")[1].split(" ")[0].split(">")[0])){
		            			test.close();
		            			continue;
		            	}
		            }
		            if(event.getEventType() == XMLEvent.END_ELEMENT){
		            	if(Arrays.asList(headers).contains(event.toString().split("::")[1].split(" ")[0].split(">")[0])){
		            			test.close();
		            			continue;
		            	}
		            }
		         
		        //avoiding start(<?xml version="1.0"?>) and end of the documents;
		        if (event.getEventType()!= XMLEvent.START_DOCUMENT && event.getEventType() != XMLEvent.END_DOCUMENT){
		        	//System.out.println(event);
		        	eventWriter.add(event);
		        	eventWriter.add(newLine);
		        	test.close();
		        }
		        	
		        }
		        
		        first = false;
		        }
		    for(String header:headers){
		    	eventWriter.add(eventFactory.createEndElement("", "", header));
			    eventWriter.add(newLine);
		    }
		    eventWriter.add(eventFactory.createEndDocument());
		    eventWriter.close();
		    return MergeFilePath;

}
}
