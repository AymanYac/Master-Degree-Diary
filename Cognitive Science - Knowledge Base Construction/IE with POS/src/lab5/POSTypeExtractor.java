package lab5;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Writer;

/**
 * Empty skeleton for a type extractor with POS tags.
 *
 * @author Fabian M. Suchanek
 */
public class POSTypeExtractor {

    /**
     * Given a POS-tagged Wikipedia article, returns the type (=class) of which
     * the article entity is an instance. For example, from a page starting with
     * "Leicester/NNP is/VBZ a/DT city/NN in/IN the/DT Midlands/NNP", you should
     * extract "city".
     * 
     * - extract the longest possible type ("American rock star")
     * consisting of adjectives and nouns, including nationalities 
     * - do not extract provenance or specifics ("from...", "in...", "by...") 
     * - do not extract too general words ("type of", "way", "form of"), but resolve like
     * a human ("A Medusa is one of two forms of certain animals" -> "animals")
     * - keep the plural 
     * - do not restrict the output to hard-coded types 
     * - in case of uncertainty, skip the article by returning NULL
     */
    public static String findType(Page nextPage) {
    	String[] taggedText = nextPage.content.split(" ");
    	String previous=null;
    	String type="";
    	String rgx="form.*|type.*|kind.*|sort.*|part.*";
    	boolean subdef=false;
    	boolean foundVerb=false;
    	for(String word:taggedText){
    		if(word.matches(".+VBZ$")) foundVerb=true;
    		if(word.matches(".+PRP.*$|.+WP.*$|.+IN$|.+TO$") && foundVerb) foundVerb=false;
    		if(!subdef && foundVerb && word.matches(".+NN.*$|.+JJ.*$")) type=type+word.split("/")[0]+" ";
    		if(subdef && foundVerb && word.matches(".+NN.*$|.")) type=type+word.split("/")[0]+" ";
    		if(word.equals("of/IN") && previous.matches(rgx)) {
    			type="";
    			foundVerb=true;
    			subdef=true;
    		}
    		previous=word.split("/")[0];
    	}
    	return (type);
    }

    /**
     * Given as arguments (1) a POS-tagged Wikipedia and (2) a target file, writes "Title
     * TAB class NEWLINE" to the target file
     */
    public static void main(String args[]) throws IOException {
        // Uncommment the following line for your convenience. Comment it out
        // again before submitting!
        //args = new String[] { "c:/fabian/data/wikipedia/wikipedia_pos.txt", "c:/fabian/data/my-output.txt" };
        //args = new String[] { "/home/moriarty/workspace/lab5/src/lab5/wikipedia-first-pos.txt", "/home/moriarty/workspace/lab5/src/lab5/my-output.txt" };
        try (Parser parser = new Parser(new File(args[0]))) {
            try (Writer result = new OutputStreamWriter(new FileOutputStream(args[1]), "UTF-8")) {
                while (parser.hasNext()) {
                    Page nextPage = parser.next();
                    String type = findType(nextPage);
                    if (type == "") continue;
                    result.write(nextPage.title + "\t" + type + "\n");
                    //System.out.println(nextPage.title + "\t" + type + "\n");
                }
            }
        }
    }

}