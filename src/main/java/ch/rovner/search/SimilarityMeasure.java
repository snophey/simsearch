package ch.rovner.search;

import java.util.List;

@FunctionalInterface
public interface SimilarityMeasure {
    /**
     * Calculates the similarity between two vectors.
     *
     * @param v1 first vector
     * @param v2 second vector
     * @return similarity score between two vectors. The greater the score, the more similar the vectors are.
     */
    Double similarity(List<Float> v1, List<Float> v2);
}
