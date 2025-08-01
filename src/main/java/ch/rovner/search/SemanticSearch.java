package ch.rovner.search;

import blogspot.software_and_algorithms.stern_library.optimization.HungarianAlgorithm;

import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class SemanticSearch {
    private final EmbeddingMapper embeddingModel;
    private final SimilarityMeasure similarityMeasure;

    /**
     * Constructs a SemanticSearch instance with the given embedding model and similarity measure.
     *
     * @param embeddingModel The embedding model to use for generating vector representations of text.
     * @param similarityMeasure The similarity measure to use for comparing vector representations.
     */
    public SemanticSearch(EmbeddingMapper embeddingModel, SimilarityMeasure similarityMeasure) {
        this.embeddingModel = embeddingModel;
        this.similarityMeasure = similarityMeasure;
    }

    /**
     * Constructs a SemanticSearch instance with the given embedding model and a default cosine similarity measure.
     *
     * @param embeddingModel The embedding model to use for generating vector representations of text.
     */
    public SemanticSearch(EmbeddingMapper embeddingModel) {
        this(embeddingModel, SemanticSearch::cosineSimilarity);
    }

    /**
     * Find the most similar item to the query in the given collection of items.
     *
     * @param query String representation of the query
     * @param items Collection of items to search in
     * @param itemToString Function to convert an item to a string representation
     * @return The most similar item to the query, if any
     * @param <T> Type of the items
     */
    public <T> Pair<T, Double> findMostSimilar(String query, Collection<T> items, Function<T, String> itemToString) {
        if (items.isEmpty()) {
            throw new IllegalArgumentException("Items must not be empty");
        }
        var mapping = getMapping(items, itemToString);
        var embeddingStore = getPopulatedStore(mapping);
        var queryEmbedding = embeddingModel.embed(query);
        return items.stream()
                .map(v -> new Pair<>(v, similarityMeasure.similarity(queryEmbedding, embeddingStore.get(itemToString.apply(v)).second())))
                .max(Comparator.comparingDouble(Pair::second))
                .get(); // we know items is not empty, so there will always be a result
    }

    /**
     * Order the items in the given collection by similarity to the query (most similar first).
     * A new list is returned, the original collection is not modified.
     *
     * @param query String representation of the query
     * @param items Collection of items to search in
     * @param itemToString Function to convert an item to a string representation
     * @return A list of items ordered by similarity to the query
     * @param <T> Type of the items
     */
    public <T> List<Pair<T, Double>> orderBySimilarity(String query, Collection<T> items, Function<T, String> itemToString) {
        var mapping = getMapping(items, itemToString);
        var embeddingStore = getPopulatedStore(mapping);
        var queryEmbedding = embeddingModel.embed(query);
        return items.stream()
                .map(v -> new Pair<>(v, similarityMeasure.similarity(queryEmbedding, embeddingStore.get(itemToString.apply(v)).second())))
                .sorted((a, b) -> -Double.compare(a.second(), b.second())) // reverse, aka descending order
                .toList();
    }

    /**
     * Pair items from two groups by semantic similarity. The groups must be of the same size.
     * The semantic similarity is calculated based on the string representation of the items.
     * So make sure that the two types U and T can be converted to strings meaningfully.
     * <p>
     * If you want to use a custom string representation, @see {@link #pairBySimilarity(List, Function, List, Function)}.
     *
     * @param group1 First group of items. This type must implement equal and hashCode methods properly since objects from this type will be used as keys in the output map
     * @param group2 Second group of items
     * @return A map pairing items from the first group to items from the second group by similarity.
     * @param <T> Type of the items in the first group
     * @param <U> Type of the items in the second group
     */
    public <T, U> Map<T, U> pairBySimilarity(List<T> group1, List<U> group2) {
        return pairBySimilarity(group1, Object::toString, group2, Object::toString);
    }

    /**
     * Pair items from two groups by semantic similarity. The groups must be of the same size.
     *
     * @param group1 First group of items. This type must implement equal and hashCode methods properly since objects from this type will be used as keys in the output map
     * @param group1ToString Function to convert an item from the first group to a string representation
     * @param group2 Second group of items
     * @param group2ToString Function to convert an item from the second group to a string representation
     * @return A map pairing items from the first group to items from the second group by similarity
     * @param <T> Type of the items in the first group
     * @param <U> Type of the items in the second group
     */
    public <T, U> Map<T, U> pairBySimilarity(List<T> group1,
                                             Function<T, String> group1ToString,
                                             List<U> group2,
                                             Function<U, String> group2ToString) {
        if (group1.size() != group2.size()) {
            throw new IllegalArgumentException("Groups must be of the same size");
        }

        Map<T, U> result = new HashMap<>();
        // build a matrix of similarities between items from the two groups, then pair them by similarity using the Hungarian algorithm
        Map<T, List<Float>> group1Embeddings = group1.stream()
                .collect(Collectors.toMap(Function.identity(), item -> embeddingModel.embed(group1ToString.apply(item))));
        Map<U, List<Float>> group2Embeddings = group2.stream()
                .collect(Collectors.toMap(Function.identity(), item -> embeddingModel.embed(group2ToString.apply(item))));

        var costMatrix = new double[group1.size()][group2.size()];
        IntStream.range(0, group1.size())
                .parallel()
                .forEach(i -> {
                    for (int j = 0; j < group2.size(); j++) {
                        // In the Hungarian algorithm, we minimize the cost. The lower the cost, the better the match.
                        // However, in our case, we have a similarity measure, where higher values mean better matches.
                        // Therefore, we negate the similarity to convert it into a cost.
                        costMatrix[i][j] = -similarityMeasure.similarity(group1Embeddings.get(group1.get(i)), group2Embeddings.get(group2.get(j)));
                    }
                });

        var assign = new HungarianAlgorithm(costMatrix).execute();
        for (int i = 0; i < assign.length; i++) {
            var j = assign[i];
            if (assign[i] != -1) {
                T item1 = group1.get(i);
                U item2 = group2.get(j);
                result.put(item1, item2);
            }
        }
        return result;
    }

    private <T> Map<String, T> getMapping(Collection<T> items, Function<T, String> itemToString) {
        return items.stream().collect(Collectors.toMap(itemToString, Function.identity()));
    }

    private <T> Map<String, Pair<T, List<Float>>> getPopulatedStore(Map<String, T> itemMapping) {
        Map<String, Pair<T, List<Float>>> embeddingStore = new HashMap<>();
        itemMapping.forEach((stringRepr, value) -> {
            var embedding = embeddingModel.embed(stringRepr);
            embeddingStore.put(stringRepr, new Pair<>(value, embedding));
        });
        return embeddingStore;
    }

    static double cosineSimilarity(List<Float> vec1, List<Float> vec2) {
        float dotProduct = 0;
        float norm1 = 0;
        float norm2 = 0;
        for (int i = 0; i < vec1.size(); i++) {
            dotProduct += vec1.get(i) * vec2.get(i);
            norm1 += vec1.get(i) * vec1.get(i);
            norm2 += vec2.get(i) * vec2.get(i);
        }
        return dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));
    }
}
