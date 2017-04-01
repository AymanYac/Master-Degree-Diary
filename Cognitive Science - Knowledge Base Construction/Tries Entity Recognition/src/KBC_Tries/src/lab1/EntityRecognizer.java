package lab1;

import java.io.File;
import java.io.IOException;
//import java.io.PrintWriter;
import java.util.Arrays;
import java.util.List;

import lab1.Trie;

/**
 * Skeleton for an entity recognizer based on a trie.
 * 
 * @author Fabian M. Suchanek
 */
public class EntityRecognizer {

    /**
     * The task is to modify the class so that it takes as arguments (1) the
     * Wikipedia corpus and (2) a file with a list of entities, and so that it
     * outputs appearances of entities in the content of articles. Each
     * appearance should be printed to the standard output as:
     * <ul>
     * <li>The title of the article where the mention occurs</li>
     * <li>TAB (\t)
     * <li>The entity mentioned</li>
     * <li>NEWLINE (\n)
     * </ul>
     * 
     * Hint: Go character by character, as in the lecture. It is not necessary
     * to go by word boundaries!
     */

    public static void main(String args[]) throws IOException {
        // Uncomment the following lines for your convenience.
        // Comment them out again before submission!
        args = new String[] { "/home/moriarty/Desktop/DK/KBC/wikipedia-first.txt","/home/moriarty/Desktop/DK/KBC/entities.txt" };
        Trie trie = new Trie(new File(args[1]));
        List<String> contentWords;
        //PrintWriter writer = new PrintWriter("/home/moriarty/Desktop/DK/KBC/out.txt", "UTF-8");
        
        
        try (Parser parser = new Parser(new File(args[0]))) {
            while (parser.hasNext()) {
                Page nextPage = parser.next();
                contentWords = Arrays.asList(nextPage.content.split(" "));
                for (int i =0; i<contentWords.size();i++){
                	String word = contentWords.get(i);
                	if(trie.search(word)){
                		System.out.println(nextPage.title+"\t"+word);
                		//writer.println(nextPage.title+"\t"+word);
                	}
                }
            }
        }
        //writer.close();
    }

}