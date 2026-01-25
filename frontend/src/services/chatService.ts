import axios from "axios";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
}

export type ChatMetrics = {
  confidenceScore: number;
  latency: number;
  accuracy: number;
  cacheHit?: boolean;
  retrievedChunks?: number;
};

// Maybe I need to modify this
// Here it is string
export interface ChatResponse {
  model_one_answer?: string;
  model_one_latency?: number;
  model_two_answer?: string;
  model_two_latency?: number;
  model_three_answer?: string;
  model_three_latency?: number;
  model_four_answer?: string;
  model_four_latency?: number;
  message?: string;
  metrics: ChatMetrics;
}

export const chatService = {
  async sendMessage(query: string): Promise<ChatResponse> {
    try {
      const startTime = Date.now();

      const response = await axios.post(
        `${API_URL}/api/chat/`,
        { query },
        {
          headers: {
            "Content-Type": "application/json",
          },
        }
      );
      // I was indeed correct
      console.log(response);

      const latency =
        response.data.latency ??
        response.data.metrics?.latency ??
        Date.now() - startTime;

      // Extract metrics from response
      const metrics: ChatMetrics = {
        confidenceScore:
          response.data.metrics?.confidence_score ||
          response.data.confidence_score ||
          0.85,
        latency,
        accuracy:
          response.data.metrics?.accuracy || response.data.accuracy || 0.9,
        retrievedChunks:
          response.data.metrics?.retrieved_chunks ||
          response.data.retrieved_chunks,
        cacheHit: response.data.metrics?.cache_hit || response.data.cache_hit,
      };

      return {
        message:
          response.data.model_two_answer ||
          response.data.response ||
          response.data.answer ||
          "Response received",
        model_one_answer: response.data.model_one_answer,

        model_two_answer: response.data.model_two_answer,

        model_three_answer: response.data.model_three_answer,
        model_four_answer: response.data.model_four_answer,
        metrics,
      };
    } catch (error: any) {
      // If backend is not available, return mock response for development
      if (error.code === "ECONNREFUSED" || error.response?.status >= 500) {
        console.warn("Backend not available, using mock response");
        return this.getResponse(query);
      }
      throw new Error(
        error.response?.data?.message ||
          error.message ||
          "Failed to send message"
      );
    }
  },

  getResponse(query: string): ChatResponse {
    // Mock response for development/testing
    const mockLatency = Math.random() * 500 + 200; // 200-700ms
    const mockConfidence = Math.random() * 0.2 + 0.75; // 0.75-0.95
    const mockAccuracy = Math.random() * 0.15 + 0.8; // 0.80-0.95

    return {
      message: `This is a mock response to your query: "${query}". The RAG system would process this query by retrieving relevant documents and generating a response based on the retrieved context.`,
      metrics: {
        confidenceScore: mockConfidence,
        latency: mockLatency,
        accuracy: mockAccuracy,
        retrievedChunks: Math.floor(Math.random() * 5) + 3,
        cacheHit: Math.random() > 0.7,
      },
    };
  },
};
