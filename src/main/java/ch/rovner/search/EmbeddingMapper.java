package ch.rovner.search;

import java.util.List;

public interface EmbeddingMapper {
    /**
     * Maps the given text to a vector representation.
     *
     * @param text The text to be mapped.
     * @return A vector representation of the text.
     */
    List<Float> embed(String text);
}
