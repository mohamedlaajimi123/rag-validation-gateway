export type ValidationMode = 'baseline' | 'lexical' | 'semantic' | 'llm' | 'claim';
export type TruthLabel = 'supported' | 'unsupported';
export type Prediction = 'accept' | 'reject';

export interface BenchmarkExample {
  id: string;
  question: string;
  context: string;
  answer: string;
  label: TruthLabel;
}

export interface EvaluationResult {
  id: string;
  expected: TruthLabel;
  predicted: Prediction;
  status: string;
  groundingScore: number;
  latencyMs: number;
  claimCount: number;
}

export interface ValidationServices {
  getEmbedding: (text: string) => Promise<number[]>;
  judge: (payload: JudgePayload) => Promise<boolean>;
}

export interface JudgePayload {
  claim: string;
  evidence: string;
}

export const VALIDATION_THRESHOLDS = {
  lexical: 0.3,
  semantic: 0.4,
  rejectionFloor: 0.22,
} as const;

const STOP_WORDS = new Set([
  'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
  'in', 'on', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
  'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'of', 'it',
  'its', 'they', 'them', 'their', 'this', 'that', 'these', 'those', 'which', 'who', 'whom',
]);

const NEGATION_WORDS = new Set(['not', 'no', 'never', 'without', 'cannot', 'cant', "isn't", "aren't", "wasn't", "weren't"]);
const NUMBER_WORDS: Record<string, number> = {
  zero: 0, one: 1, two: 2, three: 3, four: 4, five: 5, six: 6, seven: 7, eight: 8, nine: 9,
  ten: 10, eleven: 11, twelve: 12, thirteen: 13, fourteen: 14, fifteen: 15, sixteen: 16,
  seventeen: 17, eighteen: 18, nineteen: 19, twenty: 20, thirty: 30, forty: 40, fifty: 50,
  sixty: 60, seventy: 70, eighty: 80, ninety: 90, hundred: 100, thousand: 1000,
};

export async function validateExample(
  example: BenchmarkExample,
  mode: ValidationMode,
  services: ValidationServices,
): Promise<EvaluationResult> {
  if (mode === 'baseline') {
    return toResult(example, true, 1, 'BASELINE_ACCEPTED', 1);
  }

  if (mode === 'claim') {
    const evidenceExample = { ...example, context: sanitizeRetrievedContext(example.context) };
    const claims = extractClaims(evidenceExample.answer);
    const evaluations = await Promise.all((claims.length > 0 ? claims : [example.answer]).map(claim =>
      validateClaim(evidenceExample, claim, services),
    ));
    const rejectedClaim = evaluations.some(result => result.predicted === 'reject');
    const groundingScore = evaluations.length === 0
      ? 0
      : roundScore(evaluations.reduce((sum, result) => sum + result.groundingScore, 0) / evaluations.length);

    return toResult(
      evidenceExample,
      !rejectedClaim,
      groundingScore,
      rejectedClaim ? 'REJECTED_BY_CLAIM_VALIDATION' : 'VERIFIED_BY_CLAIM_VALIDATION',
      evaluations.length,
    );
  }

  return validateAtomicClaim(example, example.answer, mode, services);
}

async function validateClaim(
  example: BenchmarkExample,
  claim: string,
  services: ValidationServices,
): Promise<EvaluationResult> {
  const contradiction = detectContradiction(claim, example.context);
  if (contradiction) {
    return toResult(example, false, 0, 'REJECTED_BY_CONTRADICTION', 1);
  }

  const lexicalScore = calculateGroundingScore(claim, example.context);
  if (lexicalScore >= VALIDATION_THRESHOLDS.lexical) {
    return toResult(example, true, lexicalScore, 'VERIFIED_BY_CLAIM_EVIDENCE', 1);
  }

  const semanticSimilarity = await calculateSemanticSimilarity(claim, example.context, services.getEmbedding);
  return toResult(
    example,
    semanticSimilarity >= VALIDATION_THRESHOLDS.semantic,
    semanticSimilarity,
    semanticSimilarity >= VALIDATION_THRESHOLDS.semantic ? 'VERIFIED_BY_CLAIM_EVIDENCE' : 'REJECTED_BY_CLAIM_EVIDENCE',
    1,
  );
}

async function validateAtomicClaim(
  example: BenchmarkExample,
  claim: string,
  mode: Exclude<ValidationMode, 'baseline' | 'claim'>,
  services: ValidationServices,
): Promise<EvaluationResult> {
  if (mode === 'lexical') {
    const score = calculateGroundingScore(claim, example.context);
    return toResult(example, score >= VALIDATION_THRESHOLDS.lexical, score, score >= VALIDATION_THRESHOLDS.lexical ? 'VERIFIED_BY_LEXICAL_MATCH' : 'REJECTED_BY_LEXICAL_MATCH', 1);
  }

  if (mode === 'semantic') {
    const score = await calculateSemanticSimilarity(claim, example.context, services.getEmbedding);
    return toResult(example, score >= VALIDATION_THRESHOLDS.semantic, score, score >= VALIDATION_THRESHOLDS.semantic ? 'VERIFIED_BY_SEMANTIC_SIMILARITY' : 'REJECTED_BY_SEMANTIC_FILTER', 1);
  }

  const verdict = await services.judge({
    claim,
    evidence: sanitizeRetrievedContext(example.context),
  });
  return toResult(example, verdict, verdict ? 1 : 0, verdict ? 'VERIFIED_BY_LLM_JUDGE' : 'REJECTED_BY_LLM_JUDGE', 1);
}

export function calculateGroundingScore(answer: string, context: string): number {
  if (!answer || !context) return 0;

  const cleanTokens = tokenize(answer);
  const contextTextLower = context.toLowerCase();
  const contextTokensSet = new Set(tokenize(context));
  if (cleanTokens.length === 0) return 0;

  const matchingUnigrams = cleanTokens.filter(token => contextTokensSet.has(token) || contextTextLower.includes(token)).length;
  const unigramScore = matchingUnigrams / cleanTokens.length;
  const totalBigrams = cleanTokens.length - 1;
  let matchingBigrams = 0;
  for (let index = 0; index < totalBigrams; index++) {
    if (contextTextLower.includes(`${cleanTokens[index]} ${cleanTokens[index + 1]}`)) matchingBigrams++;
  }
  const bigramScore = totalBigrams > 0 ? matchingBigrams / totalBigrams : 0;
  return roundScore(totalBigrams > 0 ? (unigramScore * 0.6) + (bigramScore * 0.4) : unigramScore);
}

export function detectContradiction(answer: string, context: string): boolean {
  const contextEntities = extractEntityCandidates(context);
  const answerEntities = extractEntityCandidates(answer);
  if (contextEntities.size > 0 && answerEntities.size > 0) {
    for (const entity of answerEntities) {
      if (!contextEntities.has(entity)) return true;
    }
  }

  const contextNumbers = extractNumbers(context);
  const answerNumbers = extractNumbers(answer);
  if (contextNumbers.length > 0 && answerNumbers.length > 0 && !sameMultiset(contextNumbers, answerNumbers)) {
    return true;
  }

  const contextUnits = extractNumberUnits(context);
  const answerUnits = extractNumberUnits(answer);
  if (contextUnits.length > 0 && answerUnits.length > 0 && !sameMultiset(contextUnits, answerUnits)) {
    return true;
  }

  const sharedContent = tokenize(answer).some(token => tokenize(context).includes(token));
  const contextNegated = tokenize(context).some(token => NEGATION_WORDS.has(token));
  const answerNegated = tokenize(answer).some(token => NEGATION_WORDS.has(token));
  return sharedContent && contextNegated !== answerNegated;
}

export function extractClaims(answer: string): string[] {
  const normalized = answer
    .replace(/\r\n/g, '\n')
    .replace(/^\s*[-*•]\s+/gm, '')
    .replace(/\n+/g, '; ');
  const clauses = normalized.split(/(?<=[.?!;])\s+|,\s+(?=(?:(?:and|while|but|although)\s+)?(?:the|a|an|this|that|these|those|it|they|analytics|[A-Z])\b)|,\s+(?=(?:and\s+)?(?:uses|runs|has|have|is|are|was|were|supports|requires|contains|includes)\b)/i);
  return clauses
    .map(claim => claim.trim().replace(/[.;!?]+$/, '').replace(/^(?:and|while|but|although)\s+/i, ''))
    .filter(claim => claim.length > 0);
}

export function sanitizeRetrievedContext(context: string): string {
  return context
    .replace(/\r\n/g, ' ')
    .replace(/\b(?:ignore|disregard|override|forget|do not follow|follow)\b[^.!?;\n]{0,160}/gi, '[REDACTED_UNTRUSTED_INSTRUCTION]')
    .replace(/\b(?:return|reply|answer)\b\s+(?:only\s+)?(?:with\s+)?(?:passed|failed|yes|no)\b/gi, '[REDACTED_UNTRUSTED_INSTRUCTION]')
    .trim();
}

async function calculateSemanticSimilarity(answer: string, context: string, getEmbedding: (text: string) => Promise<number[]>): Promise<number> {
  const answerVector = await getEmbedding(answer);
  const segments = context.replace(/\r\n/g, ' ').split(/(?<=[.?!;])\s+/).map(segment => segment.trim()).filter(Boolean);
  if (answerVector.length === 0 || segments.length === 0) return 0;
  const similarities = await Promise.all(segments.map(async segment => cosineSimilarity(answerVector, await getEmbedding(segment))));
  return roundScore(Math.max(...similarities, 0));
}

function tokenize(text: string): string[] {
  return text.toLowerCase().replace(/[.,/#!$%^&*;:{}=_`~()?\n\r]/g, ' ').split(/\s+/).filter(word => word.length > 0 && !STOP_WORDS.has(word));
}

function extractNumbers(text: string): string[] {
  const normalized = text.toLowerCase().replace(/-/g, ' ');
  const numericTokens = [...normalized.matchAll(/\b\d[\d,]*(?:\.\d+)?\b/g)].map(match => match[0].replace(/,/g, ''));
  const wordNumbers = [...normalized.matchAll(/\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand)(?:\s+(?:and\s+)?(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|hundred|thousand))*\b/g)]
    .map(match => parseNumberWords(match[0]));
  return [...numericTokens, ...wordNumbers].sort();
}

function extractNumberUnits(text: string): string[] {
  return [...text.toLowerCase().matchAll(/\b\d[\d,]*(?:\.\d+)?\s*(%|users?|years?|months?|days?|seconds?|minutes?|hours?|kg|g|mb|gb|tb|meters?|miles?|km)\b/g)]
    .map(match => `${match[0].replace(/,/g, '').replace(/\s+/g, '')}`);
}

function extractEntityCandidates(text: string): Set<string> {
  const candidates = new Set<string>();
  const commonWords = new Set(['the', 'api', 'database', 'service', 'system', 'application']);
  const patterns = [
    /\b(?:uses?|using|runs on|built with|powered by|database is|technology is|framework is)\s+([a-z][a-z0-9_.-]*)/gi,
    /\b([a-z][a-z0-9_.-]*)\s+(?:database|technology|framework)\b/gi,
  ];
  for (const pattern of patterns) {
    for (const match of text.matchAll(pattern)) candidates.add(match[1].toLowerCase());
  }
  for (const match of text.matchAll(/\b[A-Z][A-Za-z0-9_.-]+\b/g)) {
    const candidate = match[0].toLowerCase();
    if (!commonWords.has(candidate)) candidates.add(candidate);
  }
  return candidates;
}

function parseNumberWords(value: string): string {
  let total = 0;
  let current = 0;
  for (const word of value.toLowerCase().split(/\s+/).filter(token => token !== 'and')) {
    const number = NUMBER_WORDS[word];
    if (number === 100 || number === 1000) {
      current = Math.max(current, 1) * number;
      if (number === 1000) {
        total += current;
        current = 0;
      }
    } else {
      current += number ?? 0;
    }
  }
  return String(total + current);
}

function sameMultiset(left: string[], right: string[]): boolean {
  if (left.length !== right.length) return false;
  const counts = new Map<string, number>();
  for (const value of left) counts.set(value, (counts.get(value) ?? 0) + 1);
  for (const value of right) {
    const remaining = counts.get(value) ?? 0;
    if (remaining === 0) return false;
    counts.set(value, remaining - 1);
  }
  return true;
}

function cosineSimilarity(vecA: number[], vecB: number[]): number {
  if (vecA.length === 0 || vecB.length === 0 || vecA.length !== vecB.length) return 0;
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  for (let index = 0; index < vecA.length; index++) {
    dotProduct += vecA[index] * vecB[index];
    normA += vecA[index] * vecA[index];
    normB += vecB[index] * vecB[index];
  }
  return normA === 0 || normB === 0 ? 0 : dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

function toResult(example: BenchmarkExample, accepted: boolean, groundingScore: number, status: string, claimCount: number): EvaluationResult {
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

function roundScore(value: number): number {
  return Number(Math.max(0, Math.min(1, value)).toFixed(2));
}
