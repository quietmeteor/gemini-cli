/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  CountTokensResponse,
  GenerateContentResponse,
  GenerateContentParameters,
  CountTokensParameters,
  EmbedContentResponse,
  EmbedContentParameters,
  FinishReason,
  Content,
} from '@google/genai';
import { ContentGenerator } from './contentGenerator.js';

export interface LocalModelConfig {
  endpoint: string; // e.g., 'http://localhost:11434'
  model: string; // e.g., 'gemma3:27b'
  provider: 'ollama' | 'vllm' | 'custom';
}

/**
 * Local model ContentGenerator that supports Ollama and other local model servers
 */
export class LocalContentGenerator implements ContentGenerator {
  constructor(private config: LocalModelConfig) {}

  async generateContent(
    request: GenerateContentParameters,
  ): Promise<GenerateContentResponse> {
    if (this.config.provider === 'ollama') {
      return this.generateContentOllama(request);
    }
    throw new Error(`Unsupported local model provider: ${this.config.provider}`);
  }

  async generateContentStream(
    request: GenerateContentParameters,
  ): Promise<AsyncGenerator<GenerateContentResponse>> {
    if (this.config.provider === 'ollama') {
      return this.generateContentStreamOllama(request);
    }
    throw new Error(`Unsupported local model provider: ${this.config.provider}`);
  }

  async countTokens(request: CountTokensParameters): Promise<CountTokensResponse> {
    // Most local models don't support token counting, return estimate
    const text = this.extractTextFromRequest(request);
    const estimatedTokens = Math.ceil(text.length / 4); // Rough estimate: 4 chars per token
    
    return {
      totalTokens: estimatedTokens,
    };
  }

  async embedContent(request: EmbedContentParameters): Promise<EmbedContentResponse> {
    // Local embedding support would require separate implementation
    throw new Error('Embedding not supported with local models. Use Google embedding models.');
  }

  private async generateContentOllama(
    request: GenerateContentParameters,
  ): Promise<GenerateContentResponse> {
    const prompt = this.convertToOllamaPrompt(request);
    
    console.log(`[LocalModel] Sending request to ${this.config.endpoint}/api/generate`);
    console.log(`[LocalModel] Model: ${this.config.model}`);
    console.log(`[LocalModel] Prompt length: ${prompt.length} characters`);
    console.log(`[LocalModel] Prompt preview: "${prompt.substring(0, 200)}${prompt.length > 200 ? '...' : ''}"`);
    
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 120000); // 120 second timeout (2 minutes)

    try {
      const response = await fetch(`${this.config.endpoint}/api/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: this.config.model,
          prompt: prompt,
          stream: false,
          options: {
            temperature: request.config?.temperature ?? 0.7,
            top_p: request.config?.topP ?? 0.9,
            top_k: request.config?.topK ?? 40,
          },
        }),
        signal: controller.signal,
      });
      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`Ollama API error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      
      return this.convertFromOllamaResponse(data);
    } catch (error: any) {
      clearTimeout(timeoutId);
      if (error.name === 'AbortError') {
        throw new Error(`Local model endpoint timeout after 60s: ${this.config.endpoint}/api/generate is not responding. Make sure your local model server (${this.config.provider}) is running and the model "${this.config.model}" is available.`);
      }
      console.error(`[LocalModel] Request failed:`, error);
      throw new Error(`Local model request failed: ${error.message || error}`);
    }
  }

  private async *generateContentStreamOllama(
    request: GenerateContentParameters,
  ): AsyncGenerator<GenerateContentResponse> {
    const prompt = this.convertToOllamaPrompt(request);
    
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 120000); // 120 second timeout

    try {
      const response = await fetch(`${this.config.endpoint}/api/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: this.config.model,
          prompt: prompt,
          stream: true,
          options: {
            temperature: request.config?.temperature ?? 0.7,
            top_p: request.config?.topP ?? 0.9,
            top_k: request.config?.topK ?? 40,
          },
        }),
        signal: controller.signal,
      });
      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`Ollama API error: ${response.status} ${response.statusText}`);
      }

    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('No response body from Ollama');
    }

    const decoder = new TextDecoder();
    let buffer = '';

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.trim()) {
            try {
              const data = JSON.parse(line);
              if (data.response) {
                yield this.convertFromOllamaStreamResponse(data);
              }
            } catch (e) {
              // Skip invalid JSON lines
            }
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
    } catch (error: any) {
      clearTimeout(timeoutId);
      if (error.name === 'AbortError') {
        throw new Error(`Local model endpoint timeout after 60s: ${this.config.endpoint}/api/generate is not responding. Make sure your local model server (${this.config.provider}) is running and the model "${this.config.model}" is available.`);
      }
      console.error(`[LocalModel] Request failed:`, error);
      throw new Error(`Local model request failed: ${error.message || error}`);
    }
  }

  private convertToOllamaPrompt(request: GenerateContentParameters): string {
    // Convert Gemini request format to simple prompt
    let prompt = '';
    
    // Handle ContentListUnion properly
    const contents = Array.isArray(request.contents) 
      ? request.contents.filter((c): c is Content => typeof c === 'object' && 'parts' in c)
      : typeof request.contents === 'object' && 'parts' in request.contents 
        ? [request.contents as Content]
        : [];

    for (const content of contents) {
      if (content.parts) {
        for (const part of content.parts) {
          if ('text' in part && part.text) {
            prompt += part.text + '\n';
          }
          // TODO: Handle image parts if needed
        }
      }
    }
    
    return prompt.trim();
  }

  private convertFromOllamaResponse(data: any): GenerateContentResponse {
    return {
      candidates: [
        {
          content: {
            parts: [{ text: data.response || '' }],
            role: 'model',
          },
          finishReason: data.done ? FinishReason.STOP : FinishReason.MAX_TOKENS,
          index: 0,
        },
      ],
      promptFeedback: { safetyRatings: [] },
      usageMetadata: {
        promptTokenCount: data.prompt_eval_count || 0,
        candidatesTokenCount: data.eval_count || 0,
        totalTokenCount: (data.prompt_eval_count || 0) + (data.eval_count || 0),
      },
      // Getter properties - set to undefined for manual construction
      text: undefined,
      data: undefined,
      functionCalls: undefined,
      executableCode: undefined,
      codeExecutionResult: undefined,
    };
  }

  private convertFromOllamaStreamResponse(data: any): GenerateContentResponse {
    return {
      candidates: [
        {
          content: {
            parts: [{ text: data.response || '' }],
            role: 'model',
          },
          finishReason: data.done ? FinishReason.STOP : undefined,
          index: 0,
        },
      ],
      promptFeedback: { safetyRatings: [] },
      // Getter properties - set to undefined for manual construction
      text: undefined,
      data: undefined,
      functionCalls: undefined,
      executableCode: undefined,
      codeExecutionResult: undefined,
    };
  }

  private extractTextFromRequest(request: CountTokensParameters): string {
    let text = '';
    
    // Handle ContentListUnion properly
    const contents = Array.isArray(request.contents) 
      ? request.contents.filter((c): c is Content => typeof c === 'object' && 'parts' in c)
      : typeof request.contents === 'object' && 'parts' in request.contents 
        ? [request.contents as Content]
        : [];

    for (const content of contents) {
      if (content.parts) {
        for (const part of content.parts) {
          if ('text' in part && part.text) {
            text += part.text + ' ';
          }
        }
      }
    }
    
    return text.trim();
  }
}