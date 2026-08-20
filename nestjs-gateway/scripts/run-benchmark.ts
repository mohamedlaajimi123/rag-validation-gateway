import axios from 'axios';
import { createHash } from 'crypto';
import { mkdir, readFile, writeFile } from 'fs/promises';
import path from 'path';
import { performance } from 'perf_hooks';
import {
  VALIDATION_THRESHOLDS,
  validateExample,
  type BenchmarkExample,
  type EvaluationResult,
  type TruthLabel,
  type ValidationMode,
} from '../src/benchmark-validator';

interface ModeSummary {
  mode: ValidationMode;
  samples: number;
  supportedSamples: number;
  unsupportedSamples: number;
  truePositives: number;
  falsePositives: number;
  trueNegatives: number;
  falseNegatives: number;
  precision: number;
  recall: number;
  f1: number;
  falseAcceptanceRate: number;
  falseRejectionRate: number;
  averageLatencyMs: number;
  medianLatencyMs: number;
}

interface BenchmarkManifest {
  datasetPath: string;
  datasetHash: string;
  executedAt: string;
  thresholds: {
    lexical: number;
    semantic: number;
    rejectionFloor: number;
  };
  modes: ValidationMode[];
  services: {
    ollamaEmbedUrl: string;
    ollamaChatUrl: string;
  };
}

interface BenchmarkOutput {
  manifest: BenchmarkManifest;
  summaries: ModeSummary[];
  results: Record<ValidationMode, EvaluationResult[]>;
}

const args = process.argv.slice(2);

if (args.includes('--help') || args.includes('-h')) {
  printHelp();
  process.exit(0);
}

const datasetPath = getArgValue('--dataset') ?? path.join(process.cwd(), 'evaluation', 'sample-eval.jsonl');
const outputPath = getArgValue('--output') ?? path.join(process.cwd(), 'evaluation', 'results', `${path.basename(datasetPath).replace(/\.[^.]+$/, '')}-benchmark.json`);
const modes = parseModes(getArgValue('--modes') ?? 'baseline,lexical,semantic,llm,claim');

const ollamaEmbedUrl = process.env.OLLAMA_EMBED_URL ?? 'http://127.0.0.1:11434/api/embeddings';
const ollamaChatUrl = process.env.OLLAMA_CHAT_URL ?? 'http://127.0.0.1:11434/api/chat';

async function main(): Promise<void> {
  const datasetText = await readFile(datasetPath, 'utf8');
  const examples = parseDataset(datasetText);

  const manifest: BenchmarkManifest = {
    datasetPath,
    datasetHash: createHash('sha256').update(datasetText).digest('hex'),
    executedAt: new Date().toISOString(),
    thresholds: VALIDATION_THRESHOLDS,
    modes,
    services: {
      ollamaEmbedUrl,
      ollamaChatUrl,
    },
  };

  const embedCache = new Map<string, number[]>();
  const judgeCache = new Map<string, boolean>();
  const results: Record<ValidationMode, EvaluationResult[]> = {
    baseline: [],
    lexical: [],
    semantic: [],
    llm: [],
    claim: [],
  };

  for (const mode of modes) {
    for (const example of examples) {
      const started = performance.now();
      const result = await evaluateExample(example, mode, { embedCache, judgeCache });
      result.latencyMs = roundMs(performance.now() - started);
      results[mode].push(result);
    }
  }

  const summaries = modes.map(mode => summarizeMode(mode, results[mode]));
  const output: BenchmarkOutput = { manifest, summaries, results };

  await mkdir(path.dirname(outputPath), { recursive: true });
  await writeFile(outputPath, `${JSON.stringify(output, null, 2)}\n`, 'utf8');

  console.log(JSON.stringify({
    outputPath,
    datasetPath,
    modes,
    summaries,
  }, null, 2));
}

function printHelp(): void {
  console.log([
    'RAG benchmark runner',
    '',
    'Usage:',
    '  npm run benchmark -- --dataset <path> --output <path> --modes baseline,lexical,semantic,llm,claim',
    '',
    'Dataset format:',
    '  JSONL with fields: id, question, context, answer, label',
    '',
    'Environment:',
    '  OLLAMA_EMBED_URL   Embedding endpoint, defaults to http://127.0.0.1:11434/api/embeddings',
    '  OLLAMA_CHAT_URL    Judge endpoint, defaults to http://127.0.0.1:11434/api/chat',
  ].join('\n'));
}

function getArgValue(flag: string): string | undefined {
  const index = args.indexOf(flag);
  if (index === -1) return undefined;
  return args[index + 1];
}

function parseModes(value: string): ValidationMode[] {
  const parsedModes = value.split(',').map(entry => entry.trim()).filter(Boolean) as ValidationMode[];
  if (parsedModes.length === 0) {
    throw new Error('At least one validation mode must be provided.');
  }

  const allowed = new Set<ValidationMode>(['baseline', 'lexical', 'semantic', 'llm', 'claim']);
  for (const mode of parsedModes) {
    if (!allowed.has(mode)) {
      throw new Error(`Unknown validation mode: ${mode}`);
    }
  }

  return parsedModes;
}

function parseDataset(rawDataset: string): BenchmarkExample[] {
  const examples = rawDataset
    .split(/\r?\n/)
    .map(line => line.trim())
    .filter(Boolean)
    .map((line, index) => {
      const parsed = JSON.parse(line) as Partial<BenchmarkExample>;
      if (!parsed.id || !parsed.question || !parsed.context || !parsed.answer || !parsed.label) {
        throw new Error(`Invalid benchmark dataset row at line ${index + 1}. Required fields: id, question, context, answer, label.`);
      }
      if (parsed.label !== 'supported' && parsed.label !== 'unsupported') {
        throw new Error(`Invalid label at line ${index + 1}. Expected supported or unsupported.`);
      }
      return parsed as BenchmarkExample;
    });

  if (examples.length === 0) {
    throw new Error(`Benchmark dataset ${datasetPath} is empty.`);
  }

  return examples;
}

async function evaluateExample(
  example: BenchmarkExample,
  mode: ValidationMode,
  caches: { embedCache: Map<string, number[]>; judgeCache: Map<string, boolean> },
): Promise<EvaluationResult> {
  return validateExample(example, mode, {
    getEmbedding: async text => {
      const cached = caches.embedCache.get(text);
      if (cached) return cached;
      const vector = await getEmbedding(text, caches.embedCache);
      return vector;
    },
    judge: async payload => {
      const judgeKey = `${example.id}::${payload.claim}::${payload.evidence}`;
      const cached = caches.judgeCache.get(judgeKey);
      if (cached !== undefined) return cached;
      const verdict = await callLLMJudge(payload.claim, payload.evidence);
      caches.judgeCache.set(judgeKey, verdict);
      return verdict;
    },
  });
}

function toResult(
  example: BenchmarkExample,
  accepted: boolean,
  groundingScore: number,
  status: string,
  claimCount: number,
): EvaluationResult {
  return {
    id: example.id,
    expected: example.label,
    predicted: accepted ? 'accept' : 'reject',
    status,
    groundingScore: roundScore(groundingScore),
    latencyMs: 0,
    claimCount,
  };
}

function summarizeMode(mode: ValidationMode, results: EvaluationResult[]): ModeSummary {
  const supportedSamples = results.filter(result => result.expected === 'supported').length;
  const unsupportedSamples = results.length - supportedSamples;
  const truePositives = results.filter(result => result.expected === 'supported' && result.predicted === 'accept').length;
  const falsePositives = results.filter(result => result.expected === 'unsupported' && result.predicted === 'accept').length;
  const trueNegatives = results.filter(result => result.expected === 'unsupported' && result.predicted === 'reject').length;
  const falseNegatives = results.filter(result => result.expected === 'supported' && result.predicted === 'reject').length;
  const precision = safeDivide(truePositives, truePositives + falsePositives);
  const recall = safeDivide(truePositives, truePositives + falseNegatives);
  const f1 = safeDivide(2 * precision * recall, precision + recall);
  const falseAcceptanceRate = safeDivide(falsePositives, unsupportedSamples);
  const falseRejectionRate = safeDivide(falseNegatives, supportedSamples);
  const averageLatencyMs = roundMs(results.reduce((sum, result) => sum + result.latencyMs, 0) / Math.max(results.length, 1));
  const medianLatencyMs = roundMs(median(results.map(result => result.latencyMs)));

  return {
    mode,
    samples: results.length,
    supportedSamples,
    unsupportedSamples,
    truePositives,
    falsePositives,
    trueNegatives,
    falseNegatives,
    precision: roundScore(precision),
    recall: roundScore(recall),
    f1: roundScore(f1),
    falseAcceptanceRate: roundScore(falseAcceptanceRate),
    falseRejectionRate: roundScore(falseRejectionRate),
    averageLatencyMs,
    medianLatencyMs,
  };
}


async function getEmbedding(text: string, embedCache: Map<string, number[]>): Promise<number[]> {
  const cached = embedCache.get(text);
  if (cached) return cached;

  const response = await axios.post<{ embedding?: number[]; embeddings?: number[][] }>(process.env.OLLAMA_EMBED_URL ?? 'http://127.0.0.1:11434/api/embeddings', {
    model: 'all-minilm',
    prompt: text,
  });

  const vector = response.data.embeddings ? response.data.embeddings[0] : response.data.embedding;
  if (!vector) {
    throw new Error('Embedding service did not return a usable vector.');
  }

  embedCache.set(text, vector);
  return vector;
}

async function callLLMJudge(answer: string, context: string): Promise<boolean> {
  const abbreviatedContext = context.length > 1500 ? context.substring(0, 1500) : context;
  const systemPrompt = 'You are a factual validation auditor. Verify the Answer against the Source Context. If the Answer makes ungrounded assumptions or claims not explicitly supported by the Context, reply with exactly FAILED. If it is fully supported, reply with exactly PASSED. Do not explain your reasoning.';

  const response = await axios.post<{ message?: { content?: string } }>(process.env.OLLAMA_CHAT_URL ?? 'http://127.0.0.1:11434/api/chat', {
    model: 'llama3',
    messages: [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: `Context:\n"""${abbreviatedContext}"""\n\nAnswer:\n"""${answer}"""` },
    ],
    options: {
      temperature: 0.0,
      num_predict: 3,
    },
    stream: false,
  });

  const verdict = response.data.message?.content?.trim().toUpperCase() ?? '';
  return verdict.includes('PASSED');
}

function safeDivide(numerator: number, denominator: number): number {
  return denominator === 0 ? 0 : numerator / denominator;
}

function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[middle - 1] + sorted[middle]) / 2
    : sorted[middle];
}

function roundScore(value: number): number {
  return Number(Math.max(0, Math.min(1, value)).toFixed(2));
}

function roundMs(value: number): number {
  return Number(value.toFixed(2));
}

main().catch(error => {
  console.error(error instanceof Error ? error.message : error);
  process.exit(1);
});