package ch.rovner.search;

import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2q.AllMiniLmL6V2QuantizedEmbeddingModel;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class SemanticSearchTest {

    public static class LocalEmbeddingMapper implements EmbeddingMapper {
        private final EmbeddingModel embeddingModel = new AllMiniLmL6V2QuantizedEmbeddingModel();

        @Override
        public List<Float> embed(String text) {
            return embeddingModel.embed(text).content().vectorAsList();
        }
    }

    @Test
    void findMostSimilar() {
        var choices = List.of(
                "Bern is the capital of Switzerland",
                "Berliner is a type of pastry",
                "Roses are red"
        );
        var search = new SemanticSearch(new LocalEmbeddingMapper());
        var result = search.findMostSimilar("What is the capital of Switzerland?", choices, String::toString);
        assertEquals("Bern is the capital of Switzerland", result.first());
    }

    @Test
    void orderBySimilarity() {
        var choices = List.of(
                "Bern is the capital of Switzerland",
                "Zurich is the largest city in Switzerland",
                "Babylon is an ancient city",
                "Berliner is a type of pastry"
        );
        var search = new SemanticSearch(new LocalEmbeddingMapper());
        var results = search.orderBySimilarity("What is the capital of Switzerland?", choices, String::toString);
        assertEquals(4, results.size());
        assertEquals("Bern is the capital of Switzerland", results.get(0).first());
        assertEquals("Zurich is the largest city in Switzerland", results.get(1).first());
        assertEquals("Babylon is an ancient city", results.get(2).first());
        assertEquals("Berliner is a type of pastry", results.get(3).first());
    }

    @Test
    void pairBySimilarity() {
        var questions = List.of(
                "Popular mountain range",
                "German pastry",
                "Ancient city",
                "Modern city"
        );
        var answers = List.of(
                "Basel",
                "Alps",
                "Babylon",
                "Berliner"
        );
        var search = new SemanticSearch(new LocalEmbeddingMapper());
        var pairs = search.pairBySimilarity(questions, answers);
        assertEquals(4, pairs.size());
        assertEquals("Alps", pairs.get("Popular mountain range"));
        assertEquals("Berliner", pairs.get("German pastry"));
        assertEquals("Babylon", pairs.get("Ancient city"));
        assertEquals("Basel", pairs.get("Modern city"));
    }

    @Test
    void cosineSimilarity() {
        // these two vectors are orthogonal, so cosine similarity should be close to 0
        var vector1 = List.of(1.0f, 0.0f);
        var vector2 = List.of(0.0f, 1.0f);
        assertEquals(0.0, SemanticSearch.cosineSimilarity(vector1, vector2), 0.0001);

        // these two vectors are identical, so cosine similarity should be 1
        var vector3 = List.of(1.0f, 1.0f);
        var vector4 = List.of(1.0f, 1.0f);
        assertEquals(1.0, SemanticSearch.cosineSimilarity(vector3, vector4), 0.0001);

        // these two vectors are opposite, so cosine similarity should be -1
        var vector5 = List.of(1.0f, 1.0f);
        var vector6 = List.of(-1.0f, -1.0f);
        assertEquals(-1.0, SemanticSearch.cosineSimilarity(vector5, vector6), 0.0001);

        // these two vectors are similar, so cosine similarity should be close to 1
        var embeddingModel = new LocalEmbeddingMapper();
        var vector7 = embeddingModel.embed("Roses are red");
        var vector8 = embeddingModel.embed("Roses are pink");
        var similarity = SemanticSearch.cosineSimilarity(vector7, vector8);
        assertTrue(similarity > 0.75, "Expected high similarity between similar phrases");

        // these two vectors are dissimilar, so cosine similarity should be close to 0
        var vector9 = embeddingModel.embed("Roses are red");
        var vector10 = embeddingModel.embed("Lorem ipsum dolor sit amet");
        similarity = SemanticSearch.cosineSimilarity(vector9, vector10);
        assertTrue(similarity < 0.25, "Expected low similarity between dissimilar phrases");
    }
}