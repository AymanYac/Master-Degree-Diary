package lab5;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Empty skeleton for a location extractor with POS tags.
 * 
 * @author Fabian M. Suchanek
 */
public class POSLocationExtractor {

    /**
     * Given a POS-tagged Wikipedia article, returns extract wherever possible
     * the location of a geographic entity. For example, from a page starting
     * with "Leicester/NNP is/VBZ a/DT city/NN in/IN the/DT Midlands/NNP", you
     * should extract "Midlands". 
     * 
     * Try to extract exactly the region, country, or
     * city. Do not extract locations for non-geographical entities.
     * You can also skip articles by returning NULL.
     */
    public static String findLocation(Page nextPage) {
        String[] taggedText = nextPage.content.split(" ");
        boolean candidate=false;
        for(String word:taggedText){
        	if(word.split("/")[0].toLowerCase().matches("river|sea|nation|state|lake|country|province|city|island|"
        			+ "union|kingdom|region|capital|continent|ocean")){
        		candidate=true;
        	}
        	if(candidate){
        		
        		Pattern pattern = Pattern.compile("in/IN( [a-z]+/DET){0,1}( [A-Z][a-z]+/NN.*){1,}");
        		Matcher matcher = pattern.matcher(nextPage.content);
        		if (matcher.find())
        		{
        			String type="";
        			for(String mot:matcher.group(0).split(" ")){
        				if(mot.matches(".+NNP$")){
        					type=type+mot.split("/")[0]+" ";
        				}
        			}
        			return(type);
        		}
        	}
        }
        return (null);
        //
    }

    /**
     * Given as arguments (1) a POS-tagged Wikipedia and (2) a target file,
     * writes "Title TAB location NEWLINE" to the target file
     */
    public static void main(String args[]) throws IOException {
        // Uncommment the following line for your convenience. Comment it out
        // again before submitting!
        // args = new String[] { "c:/fabian/data/wikipedia/wikipedia_pos.txt", "c:/fabian/data/my-results.txt" };
        //3args = new String[] { "/home/moriarty/workspace/lab5/src/lab5/wikipedia-first-pos.txt", "/home/moriarty/workspace/lab5/src/lab5/my-output.txt" };
        try (Parser parser = new Parser(new File(args[0]))) {
            try (Writer result = new OutputStreamWriter(new FileOutputStream(args[1]), "UTF-8")) {
                while (parser.hasNext()) {
                    Page nextPage = parser.next();
                    String type = findLocation(nextPage);
                    if (type == null) continue;
                    result.write(nextPage.title + "\t" + type + "\n");
                    //System.out.println(nextPage.title + "\t" + type + "\n");
                }
            }
        }
    }

}