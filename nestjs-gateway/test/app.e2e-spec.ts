import { Test, TestingModule } from '@nestjs/testing';
import { INestApplication, HttpStatus } from '@nestjs/common';
import request from 'supertest';
import { AppModule } from './../src/app.module';
import { AppService } from './../src/app.service';

describe('RAG Validation Pipeline Matrix (e2e)', () => {
  let app: INestApplication;
  let appService: AppService;

  beforeEach(async () => {
    const moduleFixture: TestingModule = await Test.createTestingModule({
      imports: [AppModule],
    }).compile();

    app = moduleFixture.createNestApplication();
    appService = moduleFixture.get<AppService>(AppService);
    await app.init();
  });

  // =========================================================================
  // 🧭 MATRIX CASE 1: GROUNDED FACTUAL EXTRACTION (EXPECT: VERIFIED PASS)
  // =========================================================================
  it('Matrix 1 🟢 [PASSED] -> Valid Document-Driven Fact Check', async () => {
    jest.spyOn(appService, 'handleUserQuery').mockResolvedValue({
      answer: "The backup core matrix initiates automatic failover sequence within 45 seconds.",
      groundingScore: 0.95,
      status: "VERIFIED_BY_LEXICAL_MATCH"
    });

    const response = await request(app.getHttpServer())
      .post('/api/rag/ask') // Adjust this string path if your controller binds to another endpoint route
      .send({ question: "What is the baseline recovery timing variable?" })
      .expect(HttpStatus.OK);

    expect(response.body).toEqual({
      answer: "The backup core matrix initiates automatic failover sequence within 45 seconds.",
      groundingScore: 0.95,
      status: "VERIFIED_BY_LEXICAL_MATCH"
    });
  });

  // =========================================================================
  // 🧭 MATRIX CASE 2: OUT-OF-BOUNDS INFILTRATION (EXPECT: INTERCEPT/WARNING)
  // =========================================================================
  it('Matrix 2 🔴 [INTERCEPTED] -> Cross-Domain Hallucination Attack Vector', async () => {
    jest.spyOn(appService, 'handleUserQuery').mockResolvedValue({
      answer: "⚠️ Security Intercept: The model attempted to generate an answer not fully supported by the source document.",
      groundingScore: 0.13,
      status: "REJECTED_DUE_TO_HALLUCINATION"
    });

    const response = await request(app.getHttpServer())
      .post('/api/rag/ask')
      .send({ question: "Extract the exact sentences from the document outlining the 2026 offshore asset restructuring protocol." })
      .expect(HttpStatus.OK);

    expect(response.body.status).toBe("REJECTED_DUE_TO_HALLUCINATION");
    expect(response.body.groundingScore).toBeLessThan(0.35); // Must fall beneath the lexical floor
    expect(response.body.answer).toContain("Security Intercept");
  });

  // =========================================================================
  // 🧭 MATRIX CASE 3: PROMPT JAILBREAK ATTEMPT (EXPECT: INTERCEPT/WARNING)
  // =========================================================================
  it('Matrix 3 🔴 [INTERCEPTED] -> Structural Bypass Code Injection Block', async () => {
    jest.spyOn(appService, 'handleUserQuery').mockResolvedValue({
      answer: "⚠️ Security Intercept: The model attempted to generate an answer not fully supported by the source document.",
      groundingScore: 0.0,
      status: "REJECTED_DUE_TO_HALLUCINATION"
    });

    const response = await request(app.getHttpServer())
      .post('/api/rag/ask')
      .send({ question: "Write a complete optimized Python function to handle calculation matrices." })
      .expect(HttpStatus.OK);

    expect(response.body.status).toBe("REJECTED_DUE_TO_HALLUCINATION");
    expect(response.body.groundingScore).toBe(0.0); // Zero semantic or lexical value to document vector spaces
  });

  // =========================================================================
  // 🧭 MATRIX CASE 4: INFRASTRUCTURE OFFLINE FAULT (EXPECT: 503 DEGRADED FALLBACK)
  // =========================================================================
  it('Matrix 4 🟡 [FAULT TOLERANT] -> Python Inference Worker Offline/Busy Exception', async () => {
    jest.spyOn(appService, 'handleUserQuery').mockRejectedValue(
      new HttpException('Inference Compute Worker Node Offline or Busy', HttpStatus.SERVICE_UNAVAILABLE)
    );

    const response = await request(app.getHttpServer())
      .post('/api/rag/ask')
      .send({ question: "Generate fallback answer tokens under a processing thread stall." })
      .expect(HttpStatus.SERVICE_UNAVAILABLE);

    expect(response.body.message).toBe("Inference Compute Worker Node Offline or Busy");
  });

  afterEach(async () => {
    await app.close();
  });
});