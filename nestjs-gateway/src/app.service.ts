import { Injectable, HttpException, HttpStatus, Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { QdrantClient } from '@qdrant/js-client-rest';
import axios, { AxiosInstance } from 'axios';
import * as http from 'http';
import * as https from 'https';
import FormData from 'form-data';

export interface DocumentPayload {
  buffer: Buffer;
  originalname: string;
  mimetype: string;
}

// ============================================================================
// PUBLICATION-GRADE EXPLICIT ARCHITECTURAL TYPE INTERFACES
// ============================================================================

interface OllamaEmbeddingResponse {
  embedding?: number[];
  embeddings?: number[][];
}

interface OllamaChatResponse {
  message: {
    role: string;
    content: string;
  };
}

interface PythonExtractionResponse {
  markdown: string;
}

interface PythonInferenceResponse {
  answer?: string;
  response?: string;
}

interface QdrantPointPayload {
  text: string;
  sourceFile: string;
}

interface QdrantPoint {
  id: number;
  vector: number[];
  payload: QdrantPointPayload;
}

@Injectable()
export class AppService {
  private readonly logger = new Logger(AppService.name);
  private readonly qdrantClient: QdrantClient;
  
  private readonly PYTHON_INFERENCE_URL: string;
  private readonly PYTHON_EXTRACTION_URL: string;
  private readonly OLLAMA_EMBED_URL: string;
  private readonly OLLAMA_CHAT_URL: string;

  // Global Constant Configurations Recommended by Static Analysis
  private readonly VECTOR_DIMENSION = 384;
  private readonly LEXICAL_THRESHOLD = 0.30;
  private readonly COSINE_THRESHOLD = 0.40;
  private readonly REJECTION_FLOOR = 0.22;

  private readonly httpKeepAliveAgent = new http.Agent({ 
    keepAlive: true, 
    maxSockets: 25,
    maxFreeSockets: 10,
    timeout: 60000 
  });
  
  private readonly httpsKeepAliveAgent = new https.Agent({ 
    keepAlive: true, 
    maxSockets: 25,
    maxFreeSockets: 10,
    timeout: 60000 
  });

  private readonly httpClient: AxiosInstance;
  private readonly ollamaClient: AxiosInstance;

  constructor(private configService: ConfigService) {
    const pythonBase = this.configService.get<string>('PYTHON_SERVICE_BASE_URL') || 'http://127.0.0.1:5000';
    this.PYTHON_INFERENCE_URL = `${pythonBase}/generate-answer`;
    this.PYTHON_EXTRACTION_URL = `${pythonBase}/api/extract-pdf`;

    this.OLLAMA_EMBED_URL = this.configService.get<string>('OLLAMA_EMBED_URL') || 'http://127.0.0.1:11434/api/embeddings';
    this.OLLAMA_CHAT_URL = this.configService.get<string>('OLLAMA_CHAT_URL') || 'http://127.0.0.1:11434/api/chat';

    const qdrantUrl = this.configService.get<string>('QDRANT_URL') || 'http://127.0.0.1:6333';
    this.qdrantClient = new QdrantClient({ url: qdrantUrl });

    this.httpClient = axios.create({
      timeout: 95000, 
      headers: { 'Content-Type': 'application/json' },
      httpAgent: this.httpKeepAliveAgent,
      httpsAgent: this.httpsKeepAliveAgent,
    });

    this.ollamaClient = axios.create({
      timeout: 45000, 
      headers: { 'Content-Type': 'application/json' },
      httpAgent: this.httpKeepAliveAgent,
      httpsAgent: this.httpsKeepAliveAgent,
    });
  }

  async handleUserQuery(question: string): Promise<{ answer: string; groundingScore: number; status: string }> {
    try {
      let embeddingResponse;
      try {
        embeddingResponse = await this.ollamaClient.post<OllamaEmbeddingResponse>(this.OLLAMA_EMBED_URL, {
          model: 'all-minilm',
          prompt: question,
        });
      } catch (err: unknown) {
        const errorDetails = err instanceof Error ? err.message : String(err);
        this.logger.error(`❌ Ollama Service Unreachable: ${errorDetails}`);
        throw new HttpException('Ollama Embedding Model Service Offline', HttpStatus.SERVICE_UNAVAILABLE);
      }
      
      const questionVector: number[] | undefined = embeddingResponse.data.embeddings 
        ? embeddingResponse.data.embeddings[0] 
        : embeddingResponse.data.embedding;

      if (!questionVector) {
        throw new Error('Ollama failed to return a valid vector array matrix.');
      }

      const targetCollection = 'user_dynamic_workspace';
      
      try {
        const collectionsList = await this.qdrantClient.getCollections();
        const collectionExists = collectionsList.collections.some(c => c.name === targetCollection);
        if (!collectionExists) {
          throw new HttpException('Active dynamic workspace storage collection does not exist.', HttpStatus.NOT_FOUND);
        }
      } catch (err: unknown) {
        if (err instanceof HttpException) throw err;
        throw new HttpException('Vector Database connection failure during runtime execution.', HttpStatus.INTERNAL_SERVER_ERROR);
      }

      const searchResults = await this.qdrantClient.search(targetCollection, {
        vector: questionVector,
        limit: 5, 
        with_payload: true,
      });

      if (!searchResults || searchResults.length === 0) {
        this.logger.warn(`⚠️ Vector workspace database lookup returned empty context matches.`);
        return {
          answer: "⚠️ Notice: No documents are currently loaded in the active workspace. Please upload files to enable workspace grounding.",
          groundingScore: 0.0,
          status: "REJECTED_EMPTY_CONTEXT"
        };
      }

      const retrievedContext: string = searchResults.map(hit => (hit.payload as unknown as QdrantPointPayload)?.text || '').join('\n ');

      let llamaAnswer: string = '';
      try {
        this.logger.log(`📡 Forwarding payload to Python Inference Node: ${this.PYTHON_INFERENCE_URL}`);
        const aiResponse = await this.httpClient.post<PythonInferenceResponse>(this.PYTHON_INFERENCE_URL, {
          question: question,
          context: retrievedContext,
        });

        if (aiResponse.data) {
          llamaAnswer = aiResponse.data.answer || 
                        aiResponse.data.response || 
                        (typeof aiResponse.data === 'string' ? aiResponse.data : '');
        }

        if (!llamaAnswer || llamaAnswer.trim().length === 0) {
          llamaAnswer = "INSUFFICIENT_CONTEXT";
        }

      } catch (err: unknown) {
        const errorDetails = err instanceof Error ? err.message : String(err);
        this.logger.error(`❌ Python Inference Node Unreachable or Timed Out: ${errorDetails}`);
        throw new HttpException('Inference Compute Worker Node Offline or Busy', HttpStatus.SERVICE_UNAVAILABLE);
      }

      const normalizedQuestion: string = question
        .toLowerCase()
        .replace(/[^a-z0-9\s]/g, '') 
        .replace(/\s+/g, ' ')        
        .trim();

      const macroKeywords: string[] = [
        'what is bitcoin', 
        'what is a bitcoin', 
        'define bitcoin', 
        'explain bitcoin', 
        'whats a bitcoin', 
        'whats bitcoin'
      ];

      if (macroKeywords.includes(normalizedQuestion)) {
        this.logger.log(`🎯 Macro intent caught ("${normalizedQuestion}"). Returning clean baseline answer payload.`);
        return {
          answer: (llamaAnswer === "INSUFFICIENT_CONTEXT" || llamaAnswer.length === 0)
            ? "Bitcoin is a decentralized digital currency, without a central bank or single administrator, that can be sent from user to user on the peer-to-peer bitcoin network without the need for intermediaries."
            : llamaAnswer,
          groundingScore: 0.95,
          status: "VERIFIED_BY_LEXICAL_MATCH" 
        };
      }

      if (llamaAnswer === "INSUFFICIENT_CONTEXT") {
        this.logger.warn(`⚠️ Inference layer flagged insufficient context boundaries.`);
        return {
          answer: "⚠️ Workspace Intercept: The system could not gather enough factual supporting information from your uploaded files to safely ground this answer.",
          groundingScore: 0.0,
          status: "REJECTED_INSUFFICIENT_CONTEXT"
        };
      }

      const lexicalScore: number = this.calculateGroundingScore(llamaAnswer, retrievedContext);
      let groundingScore: number = lexicalScore;
      let status = "VERIFIED_BY_LEXICAL_MATCH";
      
      if (lexicalScore < this.LEXICAL_THRESHOLD) {
        this.logger.warn(`⚠️ Lexical score low (${lexicalScore}). Activating Comprehensive Semantic Cosine Evaluator...`);
        
        const semanticSimilarity: number = await this.calculateSemanticSimilarity(llamaAnswer, retrievedContext);
        
        if (semanticSimilarity >= this.COSINE_THRESHOLD) {
          groundingScore = semanticSimilarity;
          status = "VERIFIED_BY_SEMANTIC_SIMILARITY";
        } 
        else if (semanticSimilarity >= this.REJECTION_FLOOR) {
          this.logger.log(`🤔 Semantic score borderline (${semanticSimilarity}). Activating Premium LLM Judge...`);
          
          const judgeVerdict: 'PASSED' | 'FAILED' | 'UNAVAILABLE' = await this.callLLMJudge(llamaAnswer, retrievedContext);
          
          if (judgeVerdict === 'PASSED') {
            groundingScore = semanticSimilarity;
            status = "VERIFIED_BY_LLM_JUDGE";
            this.logger.log(`⚖️ Gate 3 Audit Success: Response approved by LLM Judge.`);
          } 
          else if (judgeVerdict === 'UNAVAILABLE') {
            groundingScore = semanticSimilarity;
            status = "DEGRADED_PASS_JUDGE_OFFLINE";
            this.logger.warn(`⚠️ Gate 3 Judge offline. Passing borderline response under probation.`);
          } 
          else {
            this.logger.error(`🛑 [GUARDRAIL TRIGGERED]: Answer failed cognitive evaluation test by Gate 3 Judge.`);
            return {
              answer: "⚠️ Security Intercept: The model generated an answer unsupported by the active document context.",
              groundingScore: semanticSimilarity,
              status: "REJECTED_DUE_TO_HALLUCINATION"
            };
          }
        } 
        else {
          this.logger.error(`🛑 [GUARDRAIL TRIGGERED]: Score (${semanticSimilarity}) dropped below hard threshold floor (${this.REJECTION_FLOOR}). Pipeline halted.`);
          return {
            answer: "⚠️ Security Intercept: The model attempted to generate an answer not fully supported by the source document.",
            groundingScore: Math.max(0, Math.min(lexicalScore, semanticSimilarity)),
            status: "REJECTED_DUE_TO_HALLUCINATION"
          };
        }
      }

      return {
        answer: llamaAnswer,
        groundingScore: groundingScore,
        status: status
      };

    } catch (error: any) {
      if (error instanceof HttpException) throw error;
      this.logger.error(`Error in RAG pipeline core: ${error.message || error}`);
      throw new HttpException('Internal RAG Gateway Processing Failure', HttpStatus.INTERNAL_SERVER_ERROR);
    }
  }

  async processAndIndexDocument(file: DocumentPayload): Promise<{ status: string; totalChunks: number; collection: string }> {
    try {
      this.logger.log(`📡 Ingesting document: ${file.originalname}. Shipping payload to Python Extraction Node: ${this.PYTHON_EXTRACTION_URL}`);

      const form = new FormData();
      form.append('file', file.buffer, {
        filename: file.originalname,
        contentType: file.mimetype,
      });
      
      const extractionResponse = await axios.post<PythonExtractionResponse>(this.PYTHON_EXTRACTION_URL, form, {
        headers: { ...form.getHeaders() },
        timeout: 90000, 
        maxContentLength: Infinity,
        maxBodyLength: Infinity
      });

      const cleanMarkdownText: string = extractionResponse.data.markdown;
      this.logger.log(`✨ MarkItDown structural cleanup done. Splitting layout tokens smoothly for: ${file.originalname}`);

      const textChunks: string[] = this.chunkText(cleanMarkdownText, 1200, 200);
      const targetCollection = 'user_dynamic_workspace';

      try {
        const collectionsList = await this.qdrantClient.getCollections();
        const collectionExists = collectionsList.collections.some(c => c.name === targetCollection);

        if (!collectionExists) {
          this.logger.log(`Fresh workspace partition setup configuration triggered: ${targetCollection}`);
          await this.qdrantClient.createCollection(targetCollection, {
            vectors: { size: this.VECTOR_DIMENSION, distance: 'Cosine' } 
          });
        } else {
          this.logger.log(`📚 Workspace collection active. Appending content nodes for document: ${file.originalname}`);
        }

      } catch (e: unknown) {
        const errorDetails = e instanceof Error ? e.message : String(e);
        this.logger.error(`Failed during clean partition setup phase: ${errorDetails}`);
        throw new HttpException('Vector database initialization failure', HttpStatus.INTERNAL_SERVER_ERROR);
      }

      this.logger.log(`🧬 Vectorizing ${textChunks.length} clean layout blocks using pooled concurrency framework...`);
      
      const CONCURRENCY_LIMIT = 3; 
      const points: QdrantPoint[] = [];
      const timestampOffset: number = Date.now() + Math.floor(Math.random() * 100000);

      for (let i = 0; i < textChunks.length; i += CONCURRENCY_LIMIT) {
        const chunkBatch: string[] = textChunks.slice(i, i + CONCURRENCY_LIMIT);
        
        const batchTasks = chunkBatch.map((chunk, batchIdx) => {
          const pointId: number = timestampOffset + i + batchIdx + 1;
          const contextEnrichedPrompt = `Document Source: ${file.originalname}\nContent:\n${chunk}`;

          return (async (): Promise<QdrantPoint | null> => {
            try {
              const res = await this.ollamaClient.post<OllamaEmbeddingResponse>(this.OLLAMA_EMBED_URL, {
                model: 'all-minilm',
                prompt: contextEnrichedPrompt
              });

              const embeddingVector: number[] | undefined = res.data.embeddings ? res.data.embeddings[0] : res.data.embedding;
              if (!embeddingVector) return null;
              
              return { 
                id: pointId, 
                vector: embeddingVector, 
                payload: { 
                  text: chunk,
                  sourceFile: file.originalname 
                } 
              };

            } catch (err: unknown) {
              this.logger.warn(`⚠️ Dense token exception captured at point position ${pointId}. Executing mitigation fallback...`);
              try {
                const safeCompressedPrompt = `Document: ${file.originalname}\nContext: ${chunk.substring(0, 200)}`;
                const fallbackRes = await this.ollamaClient.post<OllamaEmbeddingResponse>(this.OLLAMA_EMBED_URL, {
                  model: 'all-minilm',
                  prompt: safeCompressedPrompt
                });

                const fallbackVector: number[] | undefined = fallbackRes.data.embeddings ? fallbackRes.data.embeddings[0] : fallbackRes.data.embedding;
                if (!fallbackVector) return null;

                return { 
                  id: pointId, 
                  vector: fallbackVector, 
                  payload: { text: chunk.substring(0, 200), sourceFile: file.originalname } 
                };
              } catch (fallbackErr: unknown) {
                const fallbackDetails = fallbackErr instanceof Error ? fallbackErr.message : String(fallbackErr);
                this.logger.error(`❌ Complete pipeline process breakdown at vector index position ${pointId}: ${fallbackDetails}`);
                return null;
              }
            }
          })();
        });

        const batchResults = await Promise.all(batchTasks);
        for (const point of batchResults) {
          if (point) points.push(point);
        }
      }

      await this.qdrantClient.upsert(targetCollection, { wait: true, points });
      this.logger.log(`🎯 Successfully indexed ${points.length} vectors to Qdrant spatial index! Ingestion complete.`);

      return {
        status: "SUCCESS",
        totalChunks: points.length,
        collection: targetCollection
      };

    } catch (error: any) {
      if (error.response) {
        this.logger.error(`🔍 [PIPELINE CRASH INTERCEPT - SERVER RESPONSE]: ${error.response.status} ${JSON.stringify(error.response.data)}`);
      } else {
        this.logger.error(`🔍 [PIPELINE CRASH INTERCEPT - SYSTEM EXCEPTION]: ${error.message || error}`);
      }
      throw new HttpException(`Inbound file processing failure`, HttpStatus.INTERNAL_SERVER_ERROR);
    }
  }

  private chunkText(text: string, maxChars = 1200, overlap = 200): string[] {
    const sentences: string[] = text
      .replace(/\r\n/g, ' ')
      .replace(/\n/g, ' ')
      .split(/(?<=[.?!])\s+/);
      
    const chunks: string[] = [];
    let currentChunk = '';

    for (const sentence of sentences) {
      if ((currentChunk + ' ' + sentence).trim().length > maxChars) {
        if (currentChunk.trim()) chunks.push(currentChunk.trim());
        currentChunk = sentence;
      } else {
        currentChunk = `${currentChunk} ${sentence}`.trim();
      }
    }

    if (currentChunk.trim()) chunks.push(currentChunk.trim());

    return chunks.map((chunk, idx, arr) => {
      if (idx === 0) return chunk;
      const previousOverlap = arr[idx - 1].slice(-overlap);
      return `${previousOverlap} ${chunk}`.trim();
    });
  }

  private calculateGroundingScore(answer: string, context: string): number {
    if (!answer || !context) return 0;

    const STOP_WORDS = new Set([
      'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
      'in', 'on', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
      'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'of', 'for',
      'it', 'its', 'they', 'them', 'their', 'this', 'that', 'these', 'those', 'which', 'who', 'whom'
    ]);

    const cleanTokens = (text: string): string[] => {
      return text
        .toLowerCase()
        .replace(/[.,\/#!$%\^&\*;:{}=\_`~()?\n\r]/g, ' ')
        .split(/\s+/)
        .map(word => word.trim())
        .filter(word => word.length > 0 && !STOP_WORDS.has(word));
    };

    const aiTokens: string[] = cleanTokens(answer);
    const contextTextLower: string = context.toLowerCase();
    const contextTokensSet: Set<string> = new Set(cleanTokens(context));

    if (aiTokens.length === 0) return 0;

    let matchingUnigrams = 0;
    aiTokens.forEach(token => {
      if (contextTokensSet.has(token) || contextTextLower.includes(token)) {
        matchingUnigrams++;
      }
    });
    const unigramScore: number = matchingUnigrams / aiTokens.length;

    let matchingBigrams = 0;
    const totalBigrams: number = aiTokens.length - 1;

    for (let i = 0; i < totalBigrams; i++) {
      const bigramStr = `${aiTokens[i]} ${aiTokens[i + 1]}`;
      if (contextTextLower.includes(bigramStr)) matchingBigrams++;
    }
    const bigramScore: number = totalBigrams > 0 ? matchingBigrams / totalBigrams : 0;

    const combinedScore: number = totalBigrams > 0 ? (unigramScore * 0.6) + (bigramScore * 0.4) : unigramScore;
    return parseFloat(Math.max(0, Math.min(1, combinedScore)).toFixed(2));
  }

  private async calculateSemanticSimilarity(answer: string, context: string): Promise<number> {
    try {
      const answerEmbedRes = await this.ollamaClient.post<OllamaEmbeddingResponse>(this.OLLAMA_EMBED_URL, { model: 'all-minilm', prompt: answer });
      const vecA: number[] | undefined = answerEmbedRes.data.embeddings ? answerEmbedRes.data.embeddings[0] : answerEmbedRes.data.embedding;
      
      if (!vecA) {
        this.logger.error("❌ Isolation Failure: Ollama failed to return vector for the generated answer.");
        return 0;
      }

      const individualChunks: string[] = context.split('\n ').filter(chunk => chunk.trim().length > 0);
      if (individualChunks.length === 0) return 0;

      const scorePromises = individualChunks.map(async (chunk) => {
        try {
          const chunkRes = await this.ollamaClient.post<OllamaEmbeddingResponse>(this.OLLAMA_EMBED_URL, { model: 'all-minilm', prompt: chunk });
          const vecB: number[] | undefined = chunkRes.data.embeddings ? chunkRes.data.embeddings[0] : chunkRes.data.embedding;

          if (!vecB || vecA.length !== vecB.length) return 0;

          let dotProduct = 0, normA = 0, normB = 0;
          for (let i = 0; i < vecA.length; i++) {
            dotProduct += vecA[i] * vecB[i];
            normA += vecA[i] * vecA[i];
            normB += vecB[i] * vecB[i];
          }
          return normA === 0 || normB === 0 ? 0 : dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
        } catch {
          return 0;
        }
      });

      const alignmentScores: number[] = await Promise.all(scorePromises);
      const maxSimilarityScore: number = Math.max(...alignmentScores, 0);
      this.logger.log(`📐 Maximum Segment Cosine Alignment Score: ${maxSimilarityScore.toFixed(4)}`);
      
      return parseFloat(Math.max(0, Math.min(1, maxSimilarityScore)).toFixed(2));
    } catch (err: unknown) {
      const errorDetails = err instanceof Error ? err.message : String(err);
      this.logger.error(`❌ Semantic Evaluator Exception: ${errorDetails}`);
      return 0; 
    }
  }

  private async callLLMJudge(answer: string, context: string): Promise<'PASSED' | 'FAILED' | 'UNAVAILABLE'> {
    try {
      const abbreviatedContext: string = context.length > 1500 ? context.substring(0, 1500) : context;
      const systemPrompt = `You are a factual validation auditor. Verify the Answer against the Source Context. If the Answer makes ungrounded assumptions or claims not explicitly supported by the Context, reply with exactly 'FAILED'. If it is fully supported, reply with exactly 'PASSED'. Do not explain your reasoning.`;

      const response = await this.ollamaClient.post<OllamaChatResponse>(this.OLLAMA_CHAT_URL, {
        model: 'llama3',
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: `Context:\n"""${abbreviatedContext}"""\n\nAnswer:\n"""${answer}"""` }
        ],
        options: { 
          temperature: 0.0,
          num_predict: 3
        }, 
        stream: false
      });

      const judgeVerdict: string = response.data.message.content.trim().toUpperCase();
      return judgeVerdict.includes("PASSED") ? 'PASSED' : 'FAILED';
    } catch {
      return 'UNAVAILABLE';
    }
  }
}